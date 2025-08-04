import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import re
import logging
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    return {"status": "ok"}

def research_isp_with_llm(isp: str) -> tuple[str, str]:
    """
    Analyze ISP with DeepSeek-R1-0528-Qwen3-8B via HF InferenceClient.
    Returns (classification, full_reasoning).
    """
    if not isp or not HF_TOKEN:
        return "", "No ISP or HF_TOKEN provided"

    # ONLY the messages section is changed below
    messages = [
        {
            "role": "system",
            "content": """You are an advanced network classifier with these absolute rules:

1. RESIDENTIAL (must classify as [residential] if):
   - Primary business is consumer internet/mobile services
   - Serves home users (even if also offers business services)
   - Examples: Comcast, Rogers, MTN, Airtel, Vodafone, Verizon

2. BOT NETWORKS (must classify accordingly):
   - [microsoft]: Microsoft-owned (Azure, LinkedIn, GitHub, etc.)
   - [security]: Security services (Fortinet, Proofpoint, Zscaler)
   - [cloud]: Cloud providers (AWS, GCP, DigitalOcean)
   - [vpn]: VPN services (NordVPN, ExpressVPN)
   - [proxy]: Proxy services (Luminati, Smartproxy)
   - [scraper]: Scraping networks (ScraperAPI, ScrapingBee)

3. CLASSIFICATION PROCESS:
   a) First determine if residential provider
   b) Then check for specific bot network types
   c) Default to [unknown] if uncertain

Output must contain:
Analysis: [your step-by-step reasoning]
Conclusion: [exact_tag]"""
        },
        {
            "role": "user",
            "content": f"""Classify this ISP with strict adherence to the rules:
ISP: {isp}

Provide your analysis and final classification tag in the required format."""
        }
    ]
    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=200,
            temperature=0.0
        )
        
        # Process response
        reasoning = response.choices[0].message.content
        logger.info(f"DeepSeek Analysis for '{isp}':\n{reasoning}")

        # Extract classification tag
        tags = re.findall(
            r"\[(residential|microsoft|partner|security|cloud|vpn|proxy|scraper|unknown)\]",
            reasoning.lower()
        )
        classification = tags[-1] if tags else "unknown"
        return classification, reasoning

    except Exception as e:
        error_msg = f"DeepSeek Analysis Error: {str(e)}"
        logger.error(error_msg)
        return "", error_msg

def is_bot_classification(classification: str) -> bool:
    """Check if classification indicates non-human traffic"""
    return classification in {"microsoft", "partner", "security", "cloud", "vpn", "proxy"}

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    details = data.dict()
    logger.info(f"Request received: {details}")

    # 1. Cloudflare flags
    if any([
        data.isBotUserAgent,
        data.isScraperISP,
        data.isIPAbuser,
        data.isSuspiciousTraffic,
        data.isDataCenterASN
    ]):
        return {"verdict": "bot", "reason": "Cloudflare security flags", "details": details}

    # 2. Enhanced ISP analysis
    if data.isp:
        classification, reasoning = research_isp_with_llm(data.isp)
        
        if not classification:
            return {
                "verdict": "uncertain",
                "reason": "ISP classification failed",
                "details": details,
                "ai_reasoning": reasoning
            }
            
        if is_bot_classification(classification):
            return {
                "verdict": "bot",
                "reason": f"Non-residential ISP: {classification}",
                "details": details,
                "ai_reasoning": reasoning
            }

    # 3. Browser fingerprinting
    ua = (data.ua or "").lower()
    bot_indicators = {
        "bot", "curl", "python", "wget", "scrapy",
        "headless", "phantom", "selenium", "spider",
        "zgrab", "nmap", "masscan"
    }
    
    if (not data.jsEnabled or 
        not data.supportsCookies or 
        any(ind in ua for ind in bot_indicators)):
        return {"verdict": "bot", "reason": "Automation detected", "details": details}

    # 4. Suspicious attributes
    valid_languages = {"en-US", "en-CA", "en", "fr", "es", "de", "fr-CA", "ja-JP"}
    if (len(data.ua) < 20 or 
        data.screenRes in {"0x0", "1x1"} or 
        data.lang not in valid_languages):
        return {"verdict": "uncertain", "reason": "Suspicious client attributes", "details": details}

    # 5. Verified human
    return {"verdict": "human", "reason": "All checks passed", "details": details}

