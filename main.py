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
        return "safe", "No ISP provided - default safe"

    messages = [
        {
            "role": "system",
            "content": """[SYSTEM RULES]
1. SAFE NETWORKS ([safe] tag):
   - Residential ISPs (Comcast, Rogers, Eastlink, MTN, Airtel)
   - Mobile carriers (Verizon, Vodafone, T-Mobile)
   - WiFi/SIM card networks

2. UNSAFE NETWORKS ([unsafe] tag):
   - Cloud providers (AWS, Azure, Google Cloud)
   - Scrapers/proxies (BrightData, Oxylabs)
   - Security platforms (Fortinet, Zscaler)

3. VERIFICATION REQUIRED ([verification] tag):
   - Only if clearly suspicious but unclassified

ANALYSIS REQUIREMENTS:
1. Compare ISP against known lists
2. Explicitly state which rule applies
3. Final line must contain: [safe|unsafe|verification]"""
        },
        {
            "role": "user",
            "content": f"Classify this ISP (respond concisely): {isp}"
        }
    ]

    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=150,  # Reduced to prevent cutoff
            temperature=0.1
        )
        
        reasoning = response.choices[0].message.content
        logger.info(f"ISP classification response: {reasoning}")

        # Robust tag extraction
        if "[safe]" in reasoning.lower():
            return "safe", reasoning
        elif "[unsafe]" in reasoning.lower():
            return "unsafe", reasoning
        return "safe", reasoning  # Default to safe instead of verification

    except Exception as e:
        logger.error(f"ISP classification failed: {str(e)}")
        return "safe", "Default safe classification"

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    details = data.dict()
    logger.info(f"Request received: {details}")

    # 1. Immediate red flags
    if any([data.isBotUserAgent, data.isScraperISP, 
           data.isIPAbuser, data.isSuspiciousTraffic,
           data.isDataCenterASN]):
        return {
            "verdict": "bot",
            "reason": "Automation flags detected",
            "details": details
        }
    
    # 2. ISP analysis (now with safe default)
    isp_classification, reasoning = research_isp_with_llm(data.isp)
    
    # 3. Browser integrity checks
    ua = (data.ua or "").lower()
    bot_indicators = {
        "bot", "curl", "python", "wget", "scrapy",
        "headless", "phantom", "selenium", "spider"
    }
    
    suspicious_browser = (
        not data.jsEnabled or 
        not data.supportsCookies or
        len(data.ua or "") < 20 or
        any(x in ua for x in bot_indicators) or
        data.screenRes in {"0x0", "1x1"}
    )
    
    # 4. Final decision
    if isp_classification == "unsafe":
        return {
            "verdict": "bot", 
            "reason": "Unsafe network",
            "details": details,
            "ai_reasoning": reasoning
        }
    elif suspicious_browser:
        return {
            "verdict": "captcha",
            "reason": "Browser verification needed",
            "details": details
        }
    
    # 5. Verified safe user
    return {
        "verdict": "user",
        "reason": "Authentic user",
        "details": details
    }
