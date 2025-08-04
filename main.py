import os
import replicate
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN  # ensure SDK sees it

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

async def research_isp_with_llm(isp: str) -> str:
    """
    Query Replicate Llama-3-70b for ISP and Microsoft association.
    Returns raw model output string.
    """
    if not isp or not REPLICATE_API_TOKEN:
        return ""
    prompt = f"""
Is "{isp}" any of the following:
- A Microsoft company, subsidiary, service, or brand?
- A Microsoft partner?
- A cloud provider, VPN, proxy, datacenter, or known scanning/bot network?
- Or is it a real residential ISP for consumers?

Please answer ONLY with one of these words (all lowercase): "microsoft", "partner", "cloud", "vpn", "proxy", "datacenter", "bot", "residential", or "unknown".

Answer:"""
    input = {
        "top_p": 0.9,
        "prompt": prompt,
        "min_tokens": 1,
        "temperature": 0.1,
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "presence_penalty": 1.15
    }
    try:
        # The Replicate SDK returns a generator for .stream() or a list for .run()
        output = replicate.run(
            "meta/meta-llama-3-70b-instruct",
            input=input
        )
        # .run() returns a list of strings; join them for full answer
        if isinstance(output, list):
            return "".join(output).strip().lower()
        return str(output).strip().lower()
    except Exception as e:
        print(f"Replicate API error: {e}")
        return ""

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    # 1. Cloudflare/worker flags always override
    if (
        data.isBotUserAgent or
        data.isScraperISP or
        data.isIPAbuser or
        data.isSuspiciousTraffic or
        data.isDataCenterASN
    ):
        return {
            "verdict": "bot",
            "reason": "Cloudflare flags indicate bot/suspicious",
            "details": data.dict()
        }

    # 2. ISP LLM Research (if ISP present)
    verdict_from_llm = None
    llm_raw = ""
    if data.isp and REPLICATE_API_TOKEN:
        llm_raw = await research_isp_with_llm(data.isp)
        # If answer contains "microsoft" or "partner", always block
        if any(word in llm_raw for word in ["microsoft", "partner", "cloud", "vpn", "proxy", "datacenter", "bot"]):
            verdict_from_llm = "bot"
        elif "residential" in llm_raw:
            verdict_from_llm = "human"
        else:
            verdict_from_llm = "uncertain"

    # Use ISP verdict if found
    if verdict_from_llm == "bot":
        return {
            "verdict": "bot",
            "reason": f'LLM says ISP "{data.isp}" = {llm_raw}',
            "details": data.dict()
        }
    elif verdict_from_llm == "uncertain":
        return {
            "verdict": "uncertain",
            "reason": f'LLM response for ISP "{data.isp}" uncertain: "{llm_raw}"',
            "details": data.dict()
        }
    # else verdict_from_llm is "human" or not present

    # 3. Browser and fingerprint heuristics (fallback)
    ua = (data.ua or "").lower()
    if (
        ("bot" in ua or "curl" in ua or "python" in ua or "wget" in ua or "scrapy" in ua or "headless" in ua)
        or not data.jsEnabled
        or not data.supportsCookies
    ):
        return {
            "verdict": "bot",
            "reason": "Bad UA or missing JS/cookie",
            "details": data.dict()
        }
    elif (
        len(data.ua) < 30
        or data.screenRes == "0x0"
        or data.lang not in ["en-US", "en", "fr", "es", "de"]
    ):
        return {
            "verdict": "uncertain",
            "reason": "Short UA, 0x0 screen or odd language",
            "details": data.dict()
        }
    else:
        return {
            "verdict": "human",
            "reason": "All heuristics passed",
            "details": data.dict()
        }
