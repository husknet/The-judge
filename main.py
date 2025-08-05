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
    If tag is [unknown], treat as [unsafe].
    """
    if not isp or not HF_TOKEN:
        return "safe", "No ISP provided - default safe [safe]"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional network risk classifier.\n"
                "Rules:\n"
                "- [safe]: Major residential ISPs and regular mobile providers.\n"
                "- [unsafe]: Anything cloud, security, scraping, VPN/proxy, Microsoft, or known bot infra.\n"
                "- [unknown]: Use if ISP is not found in public records or you are unsure or rather isp is unknown.\n"
                "- [verification]: Only use if the isp is a Major residential ISP but the provided flags still suggest it might be inaccurate.\n\n"
                "Give a 1-2 sentence reasoning. END with one tag: [safe], [unsafe], [unknown], or [verification]."
            )
        },
        {
            "role": "user",
            "content": f"Classify this ISP and explain why: {isp}"
        }
    ]

    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=150,
            temperature=0.1
        )
        
        reasoning = response.choices[0].message.content
        logger.info(f"ISP classification response: {reasoning}")

        # Extract decision tag and clean reasoning
        tags = re.findall(r"\[(safe|unsafe|unknown|verification)\]", reasoning.lower())
        classification = tags[-1] if tags else "safe"
        clean_reason = re.sub(r"\[.*?\]", "", reasoning).strip()
        
        # If the tag is [unknown], treat as [unsafe]
        if classification == "unknown":
            classification = "unsafe"
            clean_reason = f"{clean_reason} [tag changed from unknown to unsafe]"

        return classification, clean_reason

    except Exception as e:
        logger.error(f"ISP classification failed: {str(e)}")
        return "safe", "Default safe classification [safe]"

def format_decision(verdict: str, details: dict, isp_reason: str = "") -> dict:
    """Generate complete decision response with structured reasoning"""
    base_reasons = {
        "bot": {
            "title": "Automation detected",
            "details": "Cloud provider network" if details.get('isDataCenterASN') else 
                     "Bot user agent" if details.get('isBotUserAgent') else
                     "Multiple abuse flags triggered"
        },
        "captcha": {
            "title": "Verification required",
            "details": "JS/Cookies disabled" if not details.get('jsEnabled') or not details.get('supportsCookies') else
                      "Suspicious screen resolution" if details.get('screenRes') in {"0x0", "1x1"} else
                      "Unusual browser characteristics"
        },
        "user": {
            "title": "Authentic user",
            "details": "Residential network" if "comcast" in (details.get('isp') or "").lower() else
                      "All security checks passed"
        }
    }
    
    reason = base_reasons.get(verdict, {
        "title": "Unknown status",
        "details": "Needs manual review"
    })
    
    if isp_reason:
        reason['details'] = isp_reason
    
    return {
        "verdict": verdict,
        "reason": {
            "summary": reason['title'],
            "details": reason['details'],
            "decision_tag": f"[{verdict}]"  # Added structured decision tag
        },
        "details": details
    }

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    details = data.dict()
    logger.info(f"Request received: {details}")

    # 1. Check explicit abuse flags (highest priority)
    if any([data.isBotUserAgent, data.isScraperISP, 
           data.isIPAbuser, data.isSuspiciousTraffic,
           data.isDataCenterASN]):
        return format_decision("bot", details)
    
    # 2. Analyze ISP through AI classification
    isp_classification, isp_reason = research_isp_with_llm(data.isp)
    
    if isp_classification == "unsafe":
        return format_decision("bot", details, isp_reason)
    
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
    
    if suspicious_browser:
        return format_decision("captcha", details)
    
    # 4. Verified safe user
    return format_decision("user", details)
