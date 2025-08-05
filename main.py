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
    Classify ISP with strict rules and consistent responses
    Returns (classification, clean_reason)
    """
    if not isp or not HF_TOKEN:
        return "unsafe", "No ISP provided [unsafe]"

    # Pre-check known providers
    cloud_providers = ["azure", "aws", "google", "digitalocean", "oracle", "linode"]
    safe_providers = ["comcast", "verizon", "rogers", "vodafone", "mtn", "airtel"]
    
    isp_lower = isp.lower()
    if any(p in isp_lower for p in cloud_providers):
        return "unsafe", f"{isp} is cloud provider [unsafe]"
    if any(p in isp_lower for p in safe_providers):
        return "safe", f"{isp} is residential/mobile [safe]"

    messages = [
        {
            "role": "system",
            "content": """[STRICT RULES]
RESPOND WITH:
"<20 word reason> [tag]"

MUST USE [unsafe] FOR:
- Cloud (AWS/Azure/Google)
- Microsoft services
- Security platforms
- Scrapers/VPNs

MUST USE [safe] ONLY FOR:
- Residential ISPs
- Mobile carriers

ALL OTHERS: [unsafe]

Example: "Azure cloud service [unsafe]"
"""
        },
        {
            "role": "user",
            "content": f"Classify in 20 words max: {isp}"
        }
    ]

    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=30,
            temperature=0.01,
            stop=["\n"]
        )
        
        full_response = response.choices[0].message.content
        logger.info(f"ISP classification: {full_response[:50]}...")
        
        # Strict tag extraction
        tag_match = re.search(r"\[(safe|unsafe|verification)\]", full_response.lower())
        if not tag_match:
            return "unsafe", "No valid tag [unsafe]"
            
        tag = tag_match.group(1)
        reason = re.sub(r"\[.*?\]", "", full_response).strip()[:100]
        
        # Enforce unsafe default for any non-safe
        if tag != "safe":
            tag = "unsafe"
            reason = f"{reason} [enforced unsafe]"
            
        return tag, reason

    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return "unsafe", "Classification failed [unsafe]"

def format_decision(verdict: str, details: dict, isp_reason: str = "") -> dict:
    """Generate decision response with concise reasoning"""
    base_reasons = {
        "bot": {
            "summary": "Automation detected",
            "details": isp_reason if isp_reason else (
                "Cloud provider" if details.get('isDataCenterASN') else
                "Bot user agent" if details.get('isBotUserAgent') else
                "Multiple abuse flags"
            )
        },
        "captcha": {
            "summary": "Verification needed",
            "details": "JS/Cookies disabled" if not details.get('jsEnabled') or not details.get('supportsCookies') else
                      "Suspicious resolution" if details.get('screenRes') in {"0x0", "1x1"} else
                      "Unusual browser"
        },
        "user": {
            "summary": "Authentic user",
            "details": "Residential network" if "comcast" in (details.get('isp') or "").lower() else
                      "All checks passed"
        }
    }
    
    reason = base_reasons.get(verdict, {
        "summary": "Unknown",
        "details": "Manual review needed"
    })
    
    return {
        "verdict": verdict,
        "reason": {
            "summary": reason['summary'],
            "details": reason['details'],
            "decision_tag": f"[{verdict}]"
        },
        "details": {
            "ua": details.get('ua', '')[:100],
            "isp": details.get('isp', ''),
            "flags": {
                "isBot": details.get('isBotUserAgent', False),
                "isScraper": details.get('isScraperISP', False),
                "isDC": details.get('isDataCenterASN', False)
            }
        }
    }

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    details = data.dict()
    logger.info(f"Request - UA: {details.get('ua','')[:50]}...")

    # 1. Immediate red flags
    if any([data.isBotUserAgent, data.isScraperISP, 
           data.isIPAbuser, data.isSuspiciousTraffic,
           data.isDataCenterASN]):
        return format_decision("bot", details)
    
    # 2. ISP analysis
    isp_classification, isp_reason = research_isp_with_llm(data.isp)
    if isp_classification == "unsafe":
        return format_decision("bot", details, isp_reason)
    
    # 3. Browser checks
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
    
    # 4. Verified safe
    return format_decision("user", details)
