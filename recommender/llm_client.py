import requests
import time
import logging
import re
from fastapi import HTTPException
from config import LLAMA, MODEL

logger = logging.getLogger("recommender")

def call_llm(system: str, user: str) -> str:
    logger.info("call_llm: model=%s", MODEL)
    t0 = time.time()
    try:
        r = requests.post(
            f"{LLAMA}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": 400,
            },
            timeout=90,
        )
    except Exception as e:
        logger.exception("call_llm: HTTP error")
        raise
    ms = int((time.time() - t0) * 1000)
    logger.info("call_llm: status=%s latency_ms=%d", r.status_code, ms)
    if r.status_code != 200:
        logger.error("call_llm: non-200 body: %s", r.text[:500])
        raise HTTPException(r.status_code, r.text)
    try:
        content = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception("call_llm: parse error")
        raise HTTPException(500, f"LLM parse error: {e}")
    logger.debug("call_llm: content_head=%s", content[:200].replace("\n", "\\n"))
    return content

def strip_code_fences(text: str) -> str:
    return re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", text.strip())
