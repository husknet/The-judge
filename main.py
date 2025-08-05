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
    Returns (classification, full_reasoning)
    """
    if not isp or not HF_TOKEN:
        return "unsafe", "No ISP provided [unsafe]"

    messages = [
        {
            "role": "system",
            "content": """[STRICT CLASSIFICATION RULES]
You are a network classification expert. Analyze the ISP and respond with:

1. A concise 20-30 word analysis
2. Exactly one classification tag at the end: [safe], [unsafe], or [verification]

RULES:
- [safe]: ONLY for verified residential ISPs and mobile carriers
- [unsafe]: MUST use for cloud providers, datacenters, Microsoft services, security platforms, scrapers/VPNs
- [verification]: Only when ISP appears residential but needs human review

Example: "This is a Microsoft Azure cloud service [unsafe]"
"""
        },
        {
            "role": "user",
            "content": f"Classify this ISP with full reasoning: {isp}"
        }
    ]

    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=100,
            temperature=0.1
        )
        
        full_response = response.choices[0].message.content
        logger.info(f"Full AI classification response: {full_response}")
        
        # Extract the last valid tag from response
        tags = re.findall(r"\[(safe|unsafe|verification)\]", full_response.lower())
        classification = tags[-1] if tags else "unsafe"
        
        # For verification or unknown cases, default to unsafe
        if classification == "verification":
            classification = "unsafe"
            full_response = f"{full_response} [enforced unsafe]"
            
        return classification, full_response

    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return "unsafe", f"Classification failed: {str(e)} [unsafe]"

def format_decision(verdict: str, details: dict, isp_reason: str = "") -> dict:
    """Generate complete decision response with structured reasoning"""
    base_reasons = {
        "bot": {
            "summary": "Automation detected",
            "details": isp_reason if isp_reason else (
                "Cloud provider detected" if details.get('isDataCenterASN') else
                "Bot user agent detected" if details.get('isBotUserAgent') else
                "Multiple abuse flags triggered"
            )
        },
        "captcha": {
            "summary": "Verification required",
            "details": "JS/Cookies disabled" if not details.get('jsEnabled') or not details.get('supportsCookies') else
                      "Suspicious screen resolution" if details.get('screenRes') in {"0x0", "1x1"} else
                      "Unusual browser characteristics detected"
        },
        "user": {
            "summary": "Authentic user",
            "details": "Residential network verified" if "comcast" in (details.get('isp') or "").lower() else
                      "All security checks passed"
        }
    }
    
    reason = base_reasons.get(verdict, {
        "summary": "Unknown status",
        "details": "Needs manual review"
    })
    
    return {
        "verdict": verdict,
        "reason": {
            "summary": reason['summary'],
            "details": isp_reason if isp_reason else reason['details'],
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
    logger.info(f"Request received - UA: {details.get('ua','')[:50]}...")

    # 1. Immediate red flags check (highest priority)
    if any([data.isBotUserAgent, data.isScraperISP, 
           data.isIPAbuser, data.isSuspiciousTraffic,
           data.isDataCenterASN]):
        return format_decision("bot", details)
    
    # 2. ISP analysis through AI classification
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
