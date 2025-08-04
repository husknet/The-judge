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

HF_TOKEN   = os.getenv("HF_TOKEN", "")
MODEL_NAME = "google/flan-t5-xl"   # now using Flan-T5-XL for text-generation

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
    Analyze the ISP with Flan-T5-XL via HF InferenceClient.
    Returns (classification, full_reasoning).
    """
    if not isp or not HF_TOKEN:
        return "", "No ISP or HF_TOKEN provided"

    prompt = f"""
You are an internet investigator. Think step by step about whether "{isp}" is:
1. A Microsoft company/subsidiary/service.
2. A Microsoft partner.
3. An email security service (e.g. Fortinet, Proofpoint).
4. A cloud/VPN/proxy/datacenter/bot network.
5. Or a real residential ISP.

At the end, output exactly one tag in brackets:
[residential], [microsoft], [partner], [security], [cloud], [vpn], [proxy], or [unknown].
"""

    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.text_generation(
            prompt=prompt,
            model=MODEL_NAME,
            max_new_tokens=200,
            temperature=0.0
        )
        # HF InferenceClient returns a dict with "generated_text" for text_generation
        reasoning = response.get("generated_text", "")  
        if not reasoning:
            # sometimes it's returned as a list
            reasoning = "".join(response) if isinstance(response, list) else str(response)
        logger.info(f"AI Analysis for '{isp}':\n{reasoning}")

        tags = re.findall(
            r"\[(residential|microsoft|partner|security|cloud|vpn|proxy|unknown)\]",
            reasoning.lower()
        )
        classification = tags[-1] if tags else "unknown"
        return classification, reasoning

    except Exception as e:
        error_msg = f"AI Analysis Error: {e}"
        logger.error(error_msg)
        return "", error_msg

def is_bot_classification(classification: str) -> bool:
    return classification in ["microsoft", "partner", "security", "cloud", "vpn", "proxy"]

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    details = data.dict()
    logger.info(f"Incoming request: {details}")

    # 1. Cloudflare flags
    if any([
        data.isBotUserAgent, data.isScraperISP, data.isIPAbuser,
        data.isSuspiciousTraffic, data.isDataCenterASN
    ]):
        return {"verdict": "bot", "reason": "Cloudflare flags", "details": details}

    # 2. ISP analysis via Flan-T5-XL
    if data.isp:
        classification, reasoning = research_isp_with_llm(data.isp)
        if not classification:
            return {
                "verdict": "uncertain",
                "reason": "ISP analysis failed",
                "details": details,
                "ai_reasoning": reasoning
            }
        if is_bot_classification(classification):
            return {
                "verdict": "bot",
                "reason": f"ISP classified as {classification}",
                "details": details,
                "ai_reasoning": reasoning
            }

    # 3. Browser heuristics
    ua = (data.ua or "").lower()
    bot_inds = ["bot","curl","python","wget","scrapy","headless"]
    if not data.jsEnabled or not data.supportsCookies or any(b in ua for b in bot_inds):
        return {"verdict": "bot", "reason": "Browser check failed", "details": details}

    # 4. Suspicious characteristics
    valid_langs = ["en-US","en-CA","en","fr","es","de","fr-CA","ja-JP"]
    if len(data.ua) < 30 or data.screenRes == "0x0" or data.lang not in valid_langs:
        return {"verdict": "uncertain", "reason": "Suspicious characteristics", "details": details}

    # 5. Human
    return {"verdict": "human", "reason": "All checks passed", "details": details}
