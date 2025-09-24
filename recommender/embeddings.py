import requests
from fastapi import HTTPException
from typing import List
import logging
from config import OLLAMA, EMBED_MODEL

logger = logging.getLogger("recommender")

def embed_texts(texts: List[str]) -> List[List[float]]:
    logger.info("embed_texts: count=%d model=%s", len(texts), EMBED_MODEL)
    out: List[List[float]] = []
    for t in texts:
        try:
            r = requests.post(
                f"{OLLAMA}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": t},
                timeout=60,
            )
        except Exception as e:
            logger.exception("embed_texts: HTTP error to OLLAMA")
            raise HTTPException(500, f"embed_texts HTTP error: {e}")
        if r.status_code != 200:
            logger.error("embed_texts: non-200 from OLLAMA: %s", r.text[:400])
            raise HTTPException(r.status_code, r.text)
        try:
            out.append(r.json()["embedding"])
        except Exception as e:
            logger.exception("embed_texts: parse error from OLLAMA response")
            raise HTTPException(500, f"embed_texts parse error: {e}")
    return out
