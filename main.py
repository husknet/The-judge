import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import re
import logging
from huggingface_hub import InferenceClient  # Using Hugging Face for cost efficiency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN", "")  # Hugging Face token
MODEL_NAME = "tiiuae/falcon-180B"  # Efficient model for classification

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
    Dynamically analyzes ISPs with enhanced focus on:
    - Microsoft-owned networks
    - Microsoft partners
    - Security/email scanners
    Returns (classification, full_reasoning)
    """
    if not isp or not HF_TOKEN:
        return "", "No ISP or API token provided"
    
    prompt = f"""
    [INST] Analyze "{isp}" with these priorities:
    1. Is it DIRECTLY owned by Microsoft (Azure, LinkedIn, GitHub, etc.)?
    2. Is it a Microsoft strategic partner (hosting, cloud services)?
    3. Is it an email security service (Fortinet, Proofpoint, Mimecast)?
    4. Could it be a residential ISP?

    Answer format:
    - Start with "Analysis:"
    - Explain step-by-step
    - Final line MUST be: [residential], [microsoft], [partner], [security], [cloud], [vpn], [proxy], or [unknown]
    [/INST]
    """
    
    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.text_generation(
            prompt,
            model=MODEL_NAME,
            max_new_tokens=200,
            temperature=0.0
        )
        
        logger.info(f"AI Analysis for '{isp}':\n{response}")
        
        # Extract the most relevant classification
        tags = re.findall(
            r"\[(residential|microsoft|partner|security|cloud|vpn|proxy|unknown)\]",
            response.lower()
        )
        classification = tags[-1] if tags else "unknown"
        
        return classification, response
        
    except Exception as e:
        error_msg = f"AI Analysis Error: {str(e)}"
        logger.error(error_msg)
        return "", error_msg

def is_bot_classification(classification: str) -> bool:
    """Determines if a classification should be treated as a bot"""
    return classification in ["microsoft", "partner", "security", "cloud", "vpn", "proxy"]

@app.post("/ai-decision")
async def ai_decision(data: AICheckRequest):
    details = data.dict()
    logger.info(f"Incoming request: {details}")

    # 1. Cloudflare flags (unchanged)
    if any([
        data.isBotUserAgent,
        data.isScraperISP,
        data.isIPAbuser,
        data.isSuspiciousTraffic,
        data.isDataCenterASN
    ]):
        return {"verdict": "bot", "reason": "Cloudflare flags", "details": details}

    # 2. Enhanced ISP analysis
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

    # 3. Browser heuristics (unchanged)
    ua = (data.ua or "").lower()
    bot_indicators = ["bot", "curl", "python", "wget", "scrapy", "headless"]
    
    if (not data.jsEnabled or 
        not data.supportsCookies or 
        any(x in ua for x in bot_indicators)):
        return {"verdict": "bot", "reason": "Browser check failed", "details": details}

    # 4. Suspicious characteristics
    valid_languages = ["en-US", "en-CA", "en", "fr", "es", "de", "fr-CA"]
    if len(data.ua) < 30 or data.screenRes == "0x0" or data.lang not in valid_languages:
        return {"verdict": "uncertain", "reason": "Suspicious characteristics", "details": details}

    # 5. Human
    return {"verdict": "human", "reason": "All checks passed", "details": details}

