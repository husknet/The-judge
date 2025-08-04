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
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"  # conversational-only

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
    Uses the chat_completion API for Llama-3.3-70B-Instruct.
    Returns (classification, full_reasoning).
    """
    if not isp or not HF_TOKEN:
        return "", "No ISP or HF_TOKEN provided"

    client = InferenceClient(token=HF_TOKEN)
    system_msg = {
        "role": "system",
        "content": (
            "You are an internet investigator. Analyze the following ISP name:\n"
            f'"{isp}"\n\n'
            "Think step by step about whether it is:\n"
            "1. A Microsoft company/subsidiary/service\n"
            "2. A Microsoft partner\n"
            "3. An email security service (e.g. Fortinet, Proofpoint)\n"
            "4. A cloud/VPN/proxy/datacenter/bot network\n"
            "5. Or a real residential ISP\n\n"
            "At the end, output exactly one tag in brackets:\n"
            "[residential], [microsoft], [partner], [security], [cloud], [vpn], [proxy], or [unknown]."
        )
    }

    try:
        resp = client.chat_completion(
            model=MODEL_NAME,
            messages=[system_msg],
            max_new_tokens=200,
            temperature=0.0
        )
        reasoning = resp.choices[0].message.content
        logger.info(f"AI Analysis for '{isp}':\n{reasoning}")

        tags = re.findall(
            r"\[(residential|microsoft|partner|security|cloud|vpn|proxy|unknown)\]",
            reasoning.lower()
        )
        classification = tags[-1] if tags else "unknown"
        return classification, reasoning

    except Exception as e:
        error = f"AI Analysis Error: {e}"
        logger.error(error)
        return "", error
