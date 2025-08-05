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
    Returns: (classification, reason)
    Defaults to 'unsafe' if response is missing/malformed.
    """
    if not isp or not HF_TOKEN:
        return "unsafe", "No ISP or HF_TOKEN provided"

    prompt = (
        "You are a strict ISP risk classifier.\n"
        "Rules:\n"
        "- [safe]: ONLY for well-known RESIDENTIAL ISPs and mobile carriers. Examples: Comcast, Rogers, Verizon, MTN, Airtel.\n"
        "- [unsafe]: MUST USE for ANYTHING cloud, Microsoft, microsoft related services eg azure, any known proxy, VPN, scraper, business, datacenter, security, unknown, or if not 100% residential.\n"
        "- [verification]: Use ONLY if major residential, but browser/device flags suggest bot.\n"
        "Respond with a single short line: One phrase and EXACT tag at end. No extra text. Format:\n"
        "'REASON [safe|unsafe|verification]'\n"
        "Examples:\n"
        "Cloud provider, not residential [unsafe]\n"
        "ISP is Comcast [safe]\n"
        "ISP is unknown [unsafe]\n"
        "Residential, but browser suspicious [verification]\n"
    )

    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Classify this ISP: {isp}"}
            ],
            model=MODEL_NAME,
            max_tokens=20,
            temperature=0.01
        )

        response_text = response.choices[0].message.content.strip()
        match = re.match(r"^(.*)\[(safe|unsafe|verification)\]$", response_text)
        if match:
            tag = match.group(2)
            reason = match.group(1).strip()
        else:
            tag = "unsafe"
            reason = f"Malformed response: {response_text} [defaulted to unsafe]"

        return tag, reason

    except Exception as e:
        return "unsafe", f"Classification Error: {str(e)}"

def format_decision(verdict: str, details: dict, isp_reason: str = "") -> dict:
    """Generate decision response. Only debug reason if needed."""
    return {
        "verdict": verdict if verdict in {"bot", "captcha", "user"} else "bot",
        "reason": {
            "decision_tag": f"[{verdict}]",
            "details": isp_reason
        },
        "details": details
    }

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    details = data.dict()
    logger.info(f"Request received: {details}")

    # 1. Explicit abuse flags
    if any([data.isBotUserAgent, data.isScraperISP, 
            data.isIPAbuser, data.isSuspiciousTraffic,
            data.isDataCenterASN]):
        return format_decision("bot", details)
    
    # 2. ISP AI classification
    isp_classification, isp_reason = research_isp_with_llm(data.isp)
    if isp_classification == "unsafe":
        return format_decision("bot", details, isp_reason)
    elif isp_classification == "verification":
        return format_decision("captcha", details, isp_reason)
    
    # 3. Browser heuristics
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

    # 4. Safe
    return format_decision("user", details)
