import os
import replicate
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def research_isp_with_llm(isp: str) -> tuple[str, str]:
    """
    Ask Llama-3-70b to classify an ISP and return both the classification and full reasoning.
    Returns (classification, full_reasoning) tuple.
    """
    if not isp or not REPLICATE_API_TOKEN:
        return "", "No ISP or API token provided"
    
    prompt = f"""
You are an internet investigator. Analyze this ISP and classify it.
Think step by step about whether "{isp}" is:
- a Microsoft company/subsidiary/service
- a Microsoft partner
- a cloud/VPN/proxy/datacenter/bot network
- or a real residential ISP.

Show your reasoning, then on the last line output exactly one tag in square brackets:
[microsoft], [partner], [cloud], [vpn], [proxy], [datacenter], [bot], [residential], or [unknown].
"""
    try:
        output = replicate.run(
            "meta/meta-llama-3-70b-instruct",
            input={
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0.0,
                "top_p": 1.0
            }
        )
        
        # Process the output which may be a list of strings or a single string
        full_reasoning = "".join(output) if isinstance(output, list) else str(output)
        
        # Log the full AI reasoning
        logger.info(f"AI ISP Classification Reasoning for '{isp}':\n{full_reasoning}")
        
        # Find all matching tags and return the last one
        tags = re.findall(
            r"\[(microsoft|partner|cloud|vpn|proxy|datacenter|bot|residential|unknown)\]",
            full_reasoning.lower()
        )
        classification = tags[-1] if tags else "unknown"
        
        return classification, full_reasoning
    except Exception as e:
        error_msg = f"Error researching ISP: {str(e)}"
        logger.error(error_msg)
        return "", error_msg

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    details = data.dict()
    logger.info(f"Incoming request data: {details}")
    
    # 1. Cloudflare flags override (immediate bot detection)
    if any([
        data.isBotUserAgent,
        data.isScraperISP,
        data.isIPAbuser,
        data.isSuspiciousTraffic,
        data.isDataCenterASN
    ]):
        logger.info("Bot detected via Cloudflare flags")
        return {
            "verdict": "bot",
            "reason": "Cloudflare flags indicate bot",
            "details": details
        }

    # 2. ISP classification check
    if data.isp:
        isp_classification, full_reasoning = research_isp_with_llm(data.isp)
        
        if not isp_classification:  # If LLM failed
            logger.warning(f"Failed to classify ISP: {data.isp}")
            return {
                "verdict": "uncertain",
                "reason": f"Could not classify ISP '{data.isp}'",
                "details": details,
                "ai_reasoning": full_reasoning
            }
            
        if isp_classification != "residential":
            logger.info(f"Non-residential ISP detected: {data.isp} ({isp_classification})")
            return {
                "verdict": "bot",
                "reason": f'ISP "{data.isp}" classified as "{isp_classification}", not residential',
                "details": details,
                "ai_reasoning": full_reasoning
            }

    # 3. Browser & fingerprint heuristics
    ua = (data.ua or "").lower()
    bot_indicators = [
        "bot", "curl", "python", "wget", "scrapy", 
        "headless", "phantom", "selenium", "spider"
    ]
    
    if not data.jsEnabled or not data.supportsCookies:
        logger.info("Bot detected: Missing JavaScript or cookie support")
        return {
            "verdict": "bot",
            "reason": "Missing JavaScript or cookie support",
            "details": details
        }
    
    if any(indicator in ua for indicator in bot_indicators):
        logger.info(f"Bot detected via user agent: {ua}")
        return {
            "verdict": "bot",
            "reason": "Bot-like user agent detected",
            "details": details
        }

    # Screen and language checks
    valid_languages = ["en-US", "en-CA", "en", "fr", "es", "de", "fr-CA"]
    if len(data.ua) < 30 or data.screenRes == "0x0" or data.lang not in valid_languages:
        logger.info(f"Uncertain detection - suspicious UA/screen/lang: {data.ua}, {data.screenRes}, {data.lang}")
        return {
            "verdict": "uncertain",
            "reason": "Suspicious user agent, screen resolution, or language",
            "details": details
        }

    # 4. All checks passed - human
    logger.info("Request passed all checks as human")
    return {
        "verdict": "human",
        "reason": "All heuristics passed",
        "details": details
    }
