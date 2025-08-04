import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import re
import logging
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

class AICheckRequest(BaseModel):
    ua: Optional[str] = ""
    supportsCookies: Optional[bool] = None
    jsEnabled: Optional[bool] = None
    screenRes: Optional[str] = ""
    lang: Optional[str] = ""
    timezone: Optional[str] = ""
    headers: Optional[dict] = {}
    fingerprint: Optional[dict] = None
    isp: Optional[str] = ""
    isBotUserAgent: Optional[bool] = False
    isScraperISP: Optional[bool] = False
    isIPAbuser: Optional[bool] = False
    isSuspiciousTraffic: Optional[bool] = False
    isDataCenterASN: Optional[bool] = False

@app.get("/")
async def health():
    return {"status": "ok", "model": MODEL_NAME}

def research_isp_with_llm(isp: str) -> tuple[str, str]:
    """
    Enhanced ISP classification with validation to prevent false positives.
    Returns (classification, reasoning)
    """
    if not isp or not HF_TOKEN:
        logger.warning("Missing ISP or HF_TOKEN")
        return "", "Missing required parameters"

    messages = [
        {
            "role": "system",
            "content": """You are an advanced network classifier with strict rules:

1. RESIDENTIAL CLASSIFICATION ([residential]):
   - Primary business is consumer internet/mobile services
   - Serves home users (even if offers business/security services)
   - Examples: Rogers, Comcast, MTN, Airtel, Verizon, Vodafone
   - MUST classify as residential if matches these criteria

2. BOT NETWORKS (only classify as these if 100% certain):
   - [microsoft]: Microsoft-owned (Azure, LinkedIn, GitHub)
   - [security]: Dedicated security firms (Fortinet, Zscaler)
   - [cloud]: Cloud providers (AWS, Google Cloud)
   - [vpn]: Commercial VPN services (NordVPN, ExpressVPN)
   - [scraper]: Known scraping networks

3. OUTPUT REQUIREMENTS:
   - MUST include step-by-step reasoning
   - MUST conclude with: [residential|microsoft|security|cloud|vpn|scraper|unknown]
   - If reasoning suggests residential but conclusion differs, it will be overridden"""
        },
        {
            "role": "user",
            "content": f"""Classify this ISP with strict rule adherence:
ISP: {isp}

Provide analysis showing how each rule applies, then your conclusion tag."""
        }
    ]

    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=300,
            temperature=0.1,
            stop_sequences=["\n"]
        )
        
        reasoning = response.choices[0].message.content
        logger.info(f"Initial classification for '{isp}':\n{reasoning}")

        # Extract classification with validation
        conclusion_match = re.search(
            r"Conclusion:\s*\[([a-z]+)\]", 
            reasoning, 
            re.IGNORECASE
        )
        classification = conclusion_match.group(1).lower() if conclusion_match else "unknown"

        # Critical Validation: Override if reasoning contradicts conclusion
        residential_keywords = [
            "residential", "consumer", "home users", 
            "internet service", "mobile provider"
        ]
        if (any(kw in reasoning.lower() for kw in residential_keywords) and 
            classification != "residential"):
            logger.warning(f"Overriding classification to residential for {isp}")
            classification = "residential"
            reasoning += "\n[OVERRIDE: Corrected to residential based on analysis]"

        return classification, reasoning

    except Exception as e:
        error_msg = f"Classification Error: {str(e)}"
        logger.error(error_msg)
        return "", error_msg

def is_bot_classification(classification: str) -> bool:
    """Determine if classification indicates non-human traffic"""
    return classification in {
        "microsoft", "security", "cloud",
        "vpn", "scraper"  # Removed 'partner' and 'proxy' for clarity
    }

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    request_details = data.dict()
    logger.info(f"New request: {request_details}")

    # 1. Immediate Cloudflare bot flags
    cloudflare_flags = {
        "isBotUserAgent": data.isBotUserAgent,
        "isScraperISP": data.isScraperISP,
        "isIPAbuser": data.isIPAbuser,
        "isSuspiciousTraffic": data.isSuspiciousTraffic,
        "isDataCenterASN": data.isDataCenterASN
    }
    
    if any(cloudflare_flags.values()):
        logger.warning(f"Bot detected via Cloudflare flags: {cloudflare_flags}")
        return {
            "verdict": "bot",
            "reason": "Cloudflare security flags triggered",
            "details": request_details
        }

    # 2. Comprehensive ISP analysis
    if data.isp:
        classification, reasoning = research_isp_with_llm(data.isp)
        
        if not classification:
            logger.error(f"Failed to classify ISP: {data.isp}")
            return {
                "verdict": "uncertain",
                "reason": "ISP classification service unavailable",
                "details": request_details,
                "ai_reasoning": reasoning
            }
            
        if is_bot_classification(classification):
            logger.info(f"Bot network detected: {classification} for {data.isp}")
            return {
                "verdict": "bot",
                "reason": f"Classified as {classification} network",
                "details": request_details,
                "ai_reasoning": reasoning
            }

    # 3. Browser fingerprint checks
    ua = (data.ua or "").lower()
    bot_indicators = {
        "bot", "curl", "python", "wget", "scrapy",
        "headless", "phantom", "selenium", "spider",
        "zgrab", "nmap", "masscan", "automated"
    }
    
    browser_checks = {
        "no_js": not data.jsEnabled,
        "no_cookies": not data.supportsCookies,
        "bot_ua": any(ind in ua for ind in bot_indicators),
        "short_ua": len(data.ua or "") < 20
    }
    
    if any(browser_checks.values()):
        logger.warning(f"Bot detected via browser checks: {browser_checks}")
        return {
            "verdict": "bot",
            "reason": "Automation characteristics detected",
            "details": request_details
        }

    # 4. Suspicious attributes
    valid_languages = {
        "en-US", "en-CA", "en-GB", "en", 
        "fr", "es", "de", "fr-CA", "ja-JP"
    }
    
    suspicious_attributes = {
        "invalid_resolution": data.screenRes in {"0x0", "1x1"},
        "unusual_language": data.lang not in valid_languages,
        "missing_timezone": not data.timezone
    }
    
    if any(suspicious_attributes.values()):
        logger.info(f"Suspicious attributes: {suspicious_attributes}")
        return {
            "verdict": "uncertain",
            "reason": "Suspicious client characteristics",
            "details": request_details
        }

    # 5. Verified human traffic
    logger.info("Request passed all verification checks")
    return {
        "verdict": "human",
        "reason": "All checks passed",
        "details": request_details
    }
