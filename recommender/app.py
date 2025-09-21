import os
import json
import re
import uuid
import time
import logging
import traceback
import requests
from typing import List, Dict, Any, Optional, Set, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG = os.getenv("DEBUG", "0") == "1"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("recommender")

# ---- Config ----
OLLAMA = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
QDRANT = os.getenv("QDRANT_URL", "http://qdrant:6333")
QCOLL = os.getenv("QDRANT_COLLECTION", "products_poc")
LLAMA = os.getenv("LLAMA_STACK_URL", "http://llama-stack:8080")
MODEL = os.getenv("MODEL_ID", "llama3.2:3b")

# Output controls
MAX_RESULTS = int(os.getenv("RECO_MAX_RESULTS", "10"))
SCORE_THRESHOLD = float(os.getenv("RECO_SCORE_THRESHOLD", "0.01"))

app = FastAPI(title="POC Recommender", version="1.2.0")

# ---- Schemas ----
class PrefsRecommendRequest(BaseModel):
    customer_id: str
    preferences: Dict[str, List[str]]
    candidate_limit: int = 20
    top_k: int = 10                 # capped by MAX_RESULTS
    exclude_bought: bool = True

class Product(BaseModel):
    id: str   # SKU
    title: str
    description: str
    tags: List[str]

class IndexRequest(BaseModel):
    items: List[Product]

# ---- Embeddings ----
def embed_texts(texts: List[str]) -> List[List[float]]:
    logger.info("embed_texts: count=%d model=%s", len(texts), EMBED_MODEL)
    out = []
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

# ---- Qdrant helpers ----
def qdrant_search(vector: List[float], limit: int) -> List[Dict[str, Any]]:
    logger.info("qdrant_search: limit=%d", limit)
    try:
        r = requests.post(
            f"{QDRANT}/collections/{QCOLL}/points/search",
            json={"vector": vector, "limit": limit, "with_payload": True},
            timeout=30,
        )
    except Exception as e:
        logger.exception("qdrant_search: HTTP error")
        raise HTTPException(500, f"qdrant_search HTTP error: {e}")
    if r.status_code != 200:
        logger.error("qdrant_search: non-200: %s", r.text[:400])
        raise HTTPException(r.status_code, r.text)
    res = r.json().get("result", [])
    logger.info("qdrant_search: got=%d", len(res))
    return res

def qdrant_ids_for_skus(skus: List[str]) -> List[str]:
    ids = []
    for sku in skus:
        try:
            r = requests.post(
                f"{QDRANT}/collections/{QCOLL}/points/scroll",
                json={"filter": {"must": [{"key": "sku", "match": {"value": sku}}]}, "limit": 1},
                timeout=30,
            )
        except Exception as e:
            logger.exception("qdrant_ids_for_skus: HTTP error for sku=%s", sku)
            continue
        if r.status_code == 200:
            result = r.json().get("result", {}).get("points", [])
            if result:
                ids.append(result[0]["id"])
    logger.info("qdrant_ids_for_skus: requested=%d resolved=%d", len(skus), len(ids))
    return ids

def qdrant_payload_for_skus(skus: List[str]) -> Dict[str, Dict[str, Any]]:
    want = set(skus)
    out: Dict[str, Dict[str, Any]] = {}
    offset = None
    scanned = 0
    while True:
        try:
            r = requests.post(
                f"{QDRANT}/collections/{QCOLL}/points/scroll",
                json={"limit": 512, "with_payload": True, "offset": offset},
                timeout=30,
            )
        except Exception as e:
            logger.exception("qdrant_payload_for_skus: HTTP error during scroll")
            break
        if r.status_code != 200:
            logger.error("qdrant_payload_for_skus: non-200: %s", r.text[:400])
            break
        res = r.json().get("result", {})
        pts = res.get("points", [])
        scanned += len(pts)
        for p in pts:
            payload = p.get("payload") or {}
            sku = payload.get("sku")
            if sku in want and sku not in out:
                out[sku] = payload
        offset = res.get("next_page_offset")
        if not offset:
            break
    logger.info("qdrant_payload_for_skus: want=%d found=%d scanned=%d", len(skus), len(out), scanned)
    return out

def qdrant_recommend_by_items(
    positive_skus: List[str],
    negative_skus: Optional[List[str]] = None,
    limit: int = 20,
    exclude_skus: Optional[Set[str]] = None,
):
    logger.info(
        "qdrant_recommend_by_items: positives=%d negatives=%d limit=%d exclude=%d",
        len(positive_skus), len(negative_skus or []), limit, len(exclude_skus or []),
    )
    positive_qdrant_ids = qdrant_ids_for_skus(positive_skus)
    negative_qdrant_ids = qdrant_ids_for_skus(negative_skus or [])

    body: Dict[str, Any] = {
        "positive": positive_qdrant_ids,
        "limit": limit,
        "with_payload": True,
    }
    if negative_qdrant_ids:
        body["negative"] = negative_qdrant_ids
    if exclude_skus:
        body["filter"] = {"must_not": [{"key": "sku", "match": {"any": list(exclude_skus)}}]}

    try:
        r = requests.post(
            f"{QDRANT}/collections/{QCOLL}/points/recommend",
            json=body,
            timeout=30,
        )
    except Exception as e:
        logger.exception("qdrant_recommend_by_items: HTTP error")
        raise HTTPException(500, f"qdrant_recommend HTTP error: {e}")

    if r.status_code != 200:
        logger.error("qdrant_recommend_by_items: non-200: %s", r.text[:400])
        raise HTTPException(r.status_code, f"Qdrant recommend error: {r.text}")
    res = r.json().get("result", [])
    logger.info("qdrant_recommend_by_items: got=%d", len(res))
    return res

# ---- Scoring (unified 0–1) ----
CLICK_W = 0.6
CART_W  = 0.8
BOUGHT_W = 0.0
TAG_W   = 0.4   # multiplied by Jaccard(tag_candidate, tag_signals)

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def build_signal_tags(clicked: List[str], carted: List[str], bought: List[str]) -> Set[str]:
    payloads = qdrant_payload_for_skus(list({*clicked, *carted, *bought}))
    tags: Set[str] = set()
    for p in payloads.values():
        for t in p.get("tags", []) or []:
            tags.add(str(t))
    logger.info("build_signal_tags: distinct_tags=%d", len(tags))
    return tags

def score_candidate_unified(pid: str, payload: Dict[str, Any],
                            clicked: List[str], carted: List[str], bought: List[str],
                            signal_tags: Set[str]) -> Tuple[float, List[str], float, int]:
    reasons: List[str] = []
    cand_tags = set(payload.get("tags", []) or [])
    if pid in clicked:
        reasons.append("clicked")
    if pid in carted:
        reasons.append("added_to_cart")
    if pid in bought:
        reasons.append("bought")

    overlap_ratio = jaccard(cand_tags, signal_tags)
    overlap_count = len(cand_tags & signal_tags)
    if overlap_count > 0:
        reasons.append("tag_overlap")

    score = 0.0
    if pid in clicked:
        score += CLICK_W
    if pid in carted:
        score += CART_W
    if pid in bought:
        score += BOUGHT_W
    score += TAG_W * overlap_ratio
    score = max(0.0, min(1.0, score))
    return score, reasons, overlap_ratio, overlap_count

# ---- LLM ----
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
    # Log only the first 200 chars to avoid noise
    logger.debug("call_llm: content_head=%s", content[:200].replace("\n", "\\n"))
    return content

# ---- Index Endpoint ----
@app.post("/index")
def index_products(body: IndexRequest):
    logger.info("index_products: items=%d", len(body.items))
    texts = [f"{p.title} {p.description} {' '.join(p.tags)}" for p in body.items]
    vectors = embed_texts(texts)
    if not vectors:
        raise HTTPException(400, "No vectors produced for indexing")

    info = requests.get(f"{QDRANT}/collections/{QCOLL}", timeout=10)
    if info.status_code != 200:
        logger.info("index_products: creating collection size=%d distance=Cosine", len(vectors[0]))
        create = requests.put(
            f"{QDRANT}/collections/{QCOLL}",
            json={"vectors": {"size": len(vectors[0]), "distance": "Cosine"}},
            timeout=30,
        )
        if create.status_code not in (200, 201):
            logger.error("index_products: collection create failed: %s", create.text[:400])
            raise HTTPException(create.status_code, f"Qdrant create error: {create.text}")

    points = []
    for product, vec in zip(body.items, vectors):
        points.append(
            {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": {
                    "sku": product.id,
                    "title": product.title,
                    "description": product.description,
                    "tags": product.tags,
                },
            }
        )

    try:
        r = requests.put(
            f"{QDRANT}/collections/{QCOLL}/points?wait=true",
            json={"points": points},
            timeout=30,
        )
    except Exception as e:
        logger.exception("index_products: upsert HTTP error")
        raise HTTPException(500, f"Qdrant upsert HTTP error: {e}")

    if r.status_code not in (200, 202):
        logger.error("index_products: upsert non-2xx: %s", r.text[:400])
        raise HTTPException(r.status_code, f"Qdrant upsert error: {r.text}")

    logger.info("index_products: indexed=%d", len(points))
    return {"indexed": len(points)}

# ---- Recommend (LLM-assisted; unified scoring, LLM adds reasons only) ----
@app.post("/recommend/prefs_llm")
def recommend_with_prefs_llm(body: PrefsRecommendRequest):
    clicked = body.preferences.get("clicked", [])
    carted  = body.preferences.get("added_to_cart", [])
    bought  = body.preferences.get("bought", [])
    positives = list({*clicked, *carted, *bought})
    logger.info(
        "prefs_llm: user=%s clicked=%d carted=%d bought=%d cand_limit=%d top_k=%d thr=%.3f",
        body.customer_id, len(clicked), len(carted), len(bought), body.candidate_limit, body.top_k, SCORE_THRESHOLD
    )

    signal_tags = build_signal_tags(clicked, carted, bought)
    exclude_skus = set(bought) if body.exclude_bought else set()

    candidates = []
    if positives:
        candidates = qdrant_recommend_by_items(
            positive_skus=positives,
            negative_skus=[],
            limit=body.candidate_limit,
            exclude_skus=exclude_skus,
        )

    clicked_payloads = qdrant_payload_for_skus(clicked)
    carted_payloads  = qdrant_payload_for_skus(carted)
    clicked_entries  = [{"payload": p} for p in clicked_payloads.values()]
    carted_entries   = [{"payload": p} for p in carted_payloads.values()]
    candidates = carted_entries + clicked_entries + (candidates or [])
    logger.info("prefs_llm: candidates_after_inject=%d", len(candidates))

    if not candidates:
        vec = embed_texts(["diverse catalog best matches"])[0]
        candidates = qdrant_search(vec, limit=body.candidate_limit)
        logger.info("prefs_llm: fallback_candidates=%d", len(candidates))

    seen: Set[str] = set()
    scored: List[Dict[str, Any]] = []
    for r in candidates:
        payload = r.get("payload", {}) or {}
        sku = str(payload.get("sku") or r.get("id"))
        if body.exclude_bought and sku in exclude_skus:
            continue
        if sku in seen:
            continue
        seen.add(sku)

        score, reasons, overlap_ratio, overlap_count = score_candidate_unified(
            sku, payload, clicked, carted, bought, signal_tags
        )
        scored.append({
            "id": sku,
            "score": round(score, 4),
            "reasons": reasons,
            "overlap_tags_count": overlap_count,
            "overlap_tags_ratio": round(overlap_ratio, 4),
            "title": payload.get("title", ""),
            "tags": payload.get("tags", []),
        })

    logger.info("prefs_llm: scored=%d", len(scored))

    # Priority-aware sort: carted first, then clicked, then by score
    carted_set  = set(carted)
    clicked_set = set(clicked)
    scored.sort(
        key=lambda x: (
            x["id"] in carted_set,
            x["id"] in clicked_set,
            x["score"]
        ),
        reverse=True
    )

    # Filter by threshold and cap results
    max_out = min(body.top_k, MAX_RESULTS)
    filtered = [x for x in scored if x["score"] >= SCORE_THRESHOLD]
    top = filtered[:max_out]
    logger.info("prefs_llm: filtered>=%.3f -> %d; returning=%d", SCORE_THRESHOLD, len(filtered), len(top))

    # LLM: reasons only, same ids & scores
    allowed_json = json.dumps(
        [{"id": r["id"], "score": r["score"], "reasons": r["reasons"], "tags": r["tags"]} for r in top],
        ensure_ascii=False
    )
    signal_tags_list = sorted(list(signal_tags))
    system = (
        "You are a strict, deterministic product recommender.\n"
        "OUTPUT: Only valid JSON. No text. No code fences.\n"
        "Return exactly the items provided (same 'id' and 'score'); you may only adjust the 'reasons' array.\n"
        "Allowed reason labels: ['clicked','added_to_cart','bought','tag_overlap'].\n"
        "Use 'tag_overlap' ONLY if candidate tags intersect user signal tags."
    )
    user = (
        f"User signal tags: {signal_tags_list}\n\n"
        "Allowed output items (use exactly these ids and scores; do NOT change scores or add items):\n"
        f"{allowed_json}\n\n"
        'Return JSON strictly as {"recommendations":[{"id":"...","score":0-1,"reasons":[...]}, ...]}'
    )

    try:
        out = call_llm(system, user)
        out_clean = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", out.strip())
        data = json.loads(out_clean)
        items = data.get("recommendations", [])
        allowed = {r["id"]: r["score"] for r in top}
        final = []
        for it in items:
            i = str(it.get("id"))
            s = it.get("score")
            if i in allowed and abs(float(s) - allowed[i]) < 1e-6:
                final.append({"id": i, "score": s, "reasons": it.get("reasons", [])})
            if len(final) >= len(top):
                break
        # If LLM returns fewer, fill from deterministic top (no padding beyond filtered set)
        kset = {f["id"] for f in final}
        for r in top:
            if r["id"] not in kset:
                final.append({"id": r["id"], "score": r["score"], "reasons": r["reasons"]})
                if len(final) >= len(top):
                    break

        logger.info("prefs_llm: final_return=%d", len(final))
        return {"recommendations": final}
    except Exception as e:
        logger.error("prefs_llm: LLM failure -> fallback. Error=%s", e)
        logger.debug("prefs_llm: traceback:\n%s", traceback.format_exc())
        resp = {"recommendations": [{"id": r["id"], "score": r["score"], "reasons": r["reasons"]} for r in top],
                "note": "llm-fallback"}
        if DEBUG:
            resp["error"] = str(e)
        return resp

# ---- Recommend (Points-only; same unified scoring) ----
@app.post("/recommend/prefs_points")
def recommend_with_points(body: PrefsRecommendRequest):
    clicked = body.preferences.get("clicked", [])
    carted  = body.preferences.get("added_to_cart", [])
    bought  = body.preferences.get("bought", [])
    positives = list({*clicked, *carted, *bought})
    logger.info(
        "prefs_points: user=%s clicked=%d carted=%d bought=%d cand_limit=%d top_k=%d thr=%.3f",
        body.customer_id, len(clicked), len(carted), len(bought), body.candidate_limit, body.top_k, SCORE_THRESHOLD
    )

    signal_tags = build_signal_tags(clicked, carted, bought)
    exclude_skus = set(bought) if body.exclude_bought else set()

    candidates = []
    if positives:
        candidates = qdrant_recommend_by_items(
            positive_skus=positives,
            negative_skus=[],
            limit=body.candidate_limit,
            exclude_skus=exclude_skus,
        )

    clicked_payloads = qdrant_payload_for_skus(clicked)
    carted_payloads  = qdrant_payload_for_skus(carted)
    clicked_entries  = [{"payload": p} for p in clicked_payloads.values()]
    carted_entries   = [{"payload": p} for p in carted_payloads.values()]
    candidates = carted_entries + clicked_entries + (candidates or [])
    logger.info("prefs_points: candidates_after_inject=%d", len(candidates))

    if not candidates:
        vec = embed_texts(["diverse catalog best matches"])[0]
        candidates = qdrant_search(vec, body.candidate_limit)
        logger.info("prefs_points: fallback_candidates=%d", len(candidates))

    seen: Set[str] = set()
    scored: List[Dict[str, Any]] = []
    for r in candidates:
        payload = r.get("payload", {}) or {}
        sku = str(payload.get("sku") or r.get("id"))
        if body.exclude_bought and sku in exclude_skus:
            continue
        if sku in seen:
            continue
        seen.add(sku)

        score, reasons, overlap_ratio, overlap_count = score_candidate_unified(
            sku, payload, clicked, carted, bought, signal_tags
        )
        scored.append({
            "id": sku,
            "score": round(score, 4),
            "reasons": reasons,
            "overlap_tags_count": overlap_count,
            "overlap_tags_ratio": round(overlap_ratio, 4),
        })

    logger.info("prefs_points: scored=%d", len(scored))

    # Priority-aware sort, then filter by threshold and cap
    carted_set  = set(carted)
    clicked_set = set(clicked)
    scored.sort(
        key=lambda x: (
            x["id"] in carted_set,
            x["id"] in clicked_set,
            x["score"]
        ),
        reverse=True
    )

    max_out = min(body.top_k, MAX_RESULTS)
    filtered = [x for x in scored if x["score"] >= SCORE_THRESHOLD]
    top = filtered[:max_out]
    logger.info("prefs_points: filtered>=%.3f -> %d; returning=%d", SCORE_THRESHOLD, len(filtered), len(top))

    return {"recommendations": top}
