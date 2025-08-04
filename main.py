import os
import replicate
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

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

async def research_isp_with_llm(isp: str) -> str:
    """
    Ask Llama-3-70b to classify the ISP into exactly one of:
    "microsoft", "partner", "cloud", "vpn", "proxy",
    "datacenter", "bot", "residential", or "unknown".
    """
    if not isp or not REPLICATE_API_TOKEN:
        return ""
    prompt = (
        f'Is "{isp}" a Microsoft company/subsidiary/partner, '
        'a cloud/VPN/proxy/datacenter/bot network, '
        'or a real residential ISP? '
        'Answer with exactly one word (lowercase): '
        '"microsoft", "partner", "cloud", "vpn", '
        '"proxy", "datacenter", "bot", "residential", or "unknown".'
    )
    try:
        output = replicate.run(
            "meta/meta-llama-3-70b-instruct",
            input={"prompt": prompt, "max_tokens": 5, "temperature": 0.0}
        )
        text = "".join(output) if isinstance(output, list) else str(output)
        return text.strip().lower()
    except Exception:
        return ""

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    # 1. Cloudflare/worker flags override
    if any([
        data.isBotUserAgent,
        data.isScraperISP,
        data.isIPAbuser,
        data.isSuspiciousTraffic,
        data.isDataCenterASN
    ]):
        return {"verdict": "bot", "reason": "Cloudflare flags indicate bot", "details": data.dict()}

    # 2. ISP must classify as "residential" or it's a bot
    if data.isp:
        llm_raw = await research_isp_with_llm(data.isp)
        if llm_raw == "residential":
            # continue to heuristics below
            pass
        else:
            return {
                "verdict": "bot",
                "reason": f'ISP "{data.isp}" classified as "{llm_raw or "no response"}", not residential',
                "details": data.dict()
            }

    # 3. Browser & fingerprint heuristics fallback
    ua = (data.ua or "").lower()
    if (
        any(tok in ua for tok in ["bot", "curl", "python", "wget", "scrapy", "headless"])
        or not data.jsEnabled
        or not data.supportsCookies
    ):
        return {"verdict": "bot", "reason": "Bad UA or missing JS/cookie", "details": data.dict()}

    if (
        len(data.ua) < 30
        or data.screenRes == "0x0"
        or data.lang not in ["en-US", "en", "fr", "es", "de"]
    ):
        return {"verdict": "uncertain", "reason": "Short UA or odd language/screen", "details": data.dict()}

    # 4. All checks passed
    return {"verdict": "human", "reason": "All heuristics passed", "details": data.dict()}
