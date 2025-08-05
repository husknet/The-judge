import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import re
import logging
from huggingface_hub import InferenceClient

# Configure logging with increased max message length
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message).1000s'
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
    honeypotVisited: Optional[bool] = False  # <--- Added this line

@app.get("/")
async def health():
    return {"status": "ok", "model": MODEL_NAME}

def research_isp_with_llm(isp: str) -> tuple[str, str]:
    """
    Classify ISP with strict rules and consistent responses
    Returns (classification, full_reasoning)
    """
    if not isp or not HF_TOKEN:
        return "verification", "No ISP provided [verification]"

    messages = [
        {
            "role": "system",
            "content": """[STRICT CLASSIFICATION RULES]
You are a network classification expert. Analyze the ISP and respond with:

1. A concise analysis (1-2 sentences)
2. Exactly one classification tag at the end: [safe], [unsafe], or [verification]

RULES:
- [safe]: Use ONLY for major, well-known residential ISPs and mobile carriers (e.g., Comcast, BT, Eastlink, Rogers, AT&T, Telstra, Orange, T-Mobile, etc). If you are confident the ISP is primarily residential, use [safe]. Do NOT use [verification] for these.
- [unsafe]: Use for any ISP clearly identified as a cloud provider, datacenter, Microsoft, security/VPN/scraper, or not residential.
- [verification]: Use ONLY if there is NO info about the ISP or if it's impossible to determine its type after a good-faith search. Never use [verification] for well-known residential ISPs.

NEVER use multiple tags. Always commit to a single, best tag.

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
            max_tokens=300,  # Increased to allow full response
            temperature=0.1
        )
        
        full_response = response.choices[0].message.content
        
        # Log the complete response in chunks if needed
        max_log_length = 1000  # Adjust based on your logging system
        if len(full_response) > max_log_length:
            for i in range(0, len(full_response), max_log_length):
                logger.info(f"AI Response Part {i//max_log_length + 1}: {full_response[i:i+max_log_length]}")
        else:
            logger.info(f"Full AI classification response: {full_response}")
        
        # Extract the last valid tag from response
        tags = re.findall(r"\[(safe|unsafe|verification)\]", full_response.lower())
        classification = tags[-1] if tags else "verification"
            
        return classification, full_response

    except Exception as e:
        error_msg = f"Classification error: {str(e)}"
        logger.error(error_msg)
        return "verification", f"{error_msg} [verification]"

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
            "details": isp_reason if isp_reason and verdict == "captcha" else (
                "JS/Cookies disabled" if not details.get('jsEnabled') or not details.get('supportsCookies') else
                "Suspicious screen resolution" if details.get('screenRes') in {"0x0", "1x1"} else
                "Unusual browser characteristics detected"
            )
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
            "details": isp_reason if isp_reason and verdict != "captcha" else reason['details'],
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

    # 0. Honeypot visited check (absolute priority)
    if getattr(data, "honeypotVisited", False):
        return format_decision("bot", details, "Honeypot triggered by client")

    # 1. Immediate red flags check (highest priority after honeypot)
    if any([data.isBotUserAgent, data.isScraperISP, 
           data.isIPAbuser, data.isSuspiciousTraffic,
           data.isDataCenterASN]):
        return format_decision("bot", details)
    
    # 2. ISP analysis through AI classification
    isp_classification, isp_reason = research_isp_with_llm(data.isp)
    
    if isp_classification == "unsafe":
        return format_decision("bot", details, isp_reason)
    elif isp_classification == "verification":
        return format_decision("captcha", details, isp_reason)
    
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
