import os
import replicate
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Ensure your token is available to the Replicate SDK
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN or ""

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
    Ask Llama-3-70b to classify the ISP in one of:
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
            input={
                "prompt": prompt,
                "max_tokens": 5,
                "temperature": 0.0,
            }
        )
        # replicate.run may return a list of strings or a single string
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
        return {
            "verdict": "bot",
            "reason": "Cloudflare flags indicate bot/suspicious",
            "details": data.dict()
        }

    # 2. ISP classification via LLM
    llm_raw = await research_isp_with_llm(data.isp)
    verdict_from_llm = None

    if llm_raw in ["microsoft", "partner", "cloud", "vpn", "proxy", "datacenter", "bot"]:
        verdict_from_llm = "bot"
    elif llm_raw == "residential":
        verdict_from_llm = "human"
    elif llm_raw == "unknown":
        verdict_from_llm = "uncertain"
    # if llm_raw == "", skip to heuristics

    if verdict_from_llm == "bot":
        return {
            "verdict": "bot",
            "reason": f'LLM says ISP "{data.isp}" = {llm_raw}',
            "details": data.dict()
        }
    if verdict_from_llm == "uncertain":
        return {
            "verdict": "uncertain",
            "reason": f'LLM response for ISP "{data.isp}" uncertain: "{llm_raw}"',
            "details": data.dict()
        }

    # 3. Browser & fingerprint heuristics fallback
    ua = (data.ua or "").lower()
    if (
        any(tok in ua for tok in ["bot","curl","python","wget","scrapy","headless"])
        or not data.jsEnabled
        or not data.supportsCookies
    ):
        return {
            "verdict": "bot",
            "reason": "Bad UA or missing JS/cookie",
            "details": data.dict()
        }
    if (
        len(data.ua) < 30
        or data.screenRes == "0x0"
        or data.lang not in ["en-US","en","fr","es","de"]
    ):
        return {
            "verdict": "uncertain",
            "reason": "Short UA or odd language/screen",
            "details": data.dict()
        }

    # 4. All checks passed
    return {
        "verdict": "human",
        "reason": "All heuristics passed",
        "details": data.dict()
    }
