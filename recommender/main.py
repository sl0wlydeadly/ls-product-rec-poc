import logging
from fastapi import FastAPI, HTTPException
from typing import List
from config import LOG_LEVEL
from schemas import IndexRequest, PrefsRecommendRequest
from embeddings import embed_texts
from qdrant import ensure_collection, upsert_points, make_points
from recommender_llm import build_suggestions
from recommender_points import recommend_points

app = FastAPI(title="POC Recommender", version="1.7.2")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("recommender")

# ---- Index Endpoint ----
@app.post("/index")
def index_products(body: IndexRequest):
    logger.info("index_products: items=%d", len(body.items))
    texts = [f"{p.title} {p.description} {' '.join(p.tags)}" for p in body.items]
    vectors = embed_texts(texts)
    if not vectors:
        raise HTTPException(400, "No vectors produced for indexing")
    ensure_collection(len(vectors[0]))
    items = [p.dict() for p in body.items]
    points = make_points(items, vectors)
    upsert_points(points)
    logger.info("index_products: indexed=%d", len(points))
    return {"indexed": len(points)}

# ---- Recommend (LLM suggestions from Qdrant) ----
@app.post("/recommend/prefs_llm")
def recommend_with_prefs_llm(body: PrefsRecommendRequest):
    return build_suggestions(body)

# ---- Recommend (Points-only; unified scoring) ----
@app.post("/recommend/prefs_points")
def recommend_with_points(body: PrefsRecommendRequest):
    return recommend_points(body)
