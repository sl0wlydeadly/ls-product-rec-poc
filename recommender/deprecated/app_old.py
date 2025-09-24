""" import os
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

app = FastAPI(title="POC Recommender", version="1.7.1")

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

# ---- Recommend (LLM suggestions from Qdrant candidates; human-friendly CTA) ----
@app.post("/recommend/prefs_llm")
def recommend_with_prefs_llm(body: PrefsRecommendRequest):
    """
""" Build suggestions using items from the vector DB:
      - Use clicked/carted/bought as POSITIVE examples to Qdrant /recommend.
      - Exclude clicked and added_to_cart from TARGETS; also exclude bought if exclude_bought=True.
      - Let the LLM pick which TARGET from Qdrant options per source (we validate).
      - Server assembles human-friendly CTA sentences prompting the user to view the target.
    Returns: { customer_id, suggestions: [{text, source_sku, target_sku}] }
    """
"""
    clicked = body.preferences.get("clicked", []) or []
    carted  = body.preferences.get("added_to_cart", []) or []
    bought  = body.preferences.get("bought", []) or []

    logger.info(
        "prefs_llm(Qdrant-targets+CTA): user=%s clicked=%d carted=%d bought=%d top_k=%d exclude_bought=%s",
        body.customer_id, len(clicked), len(carted), len(bought), body.top_k, body.exclude_bought
    )

    positives = list({*clicked, *carted, *bought})
    exclude_targets: Set[str] = set(clicked) | set(carted)
    if body.exclude_bought:
        exclude_targets |= set(bought)

    candidates: List[Dict[str, Any]] = []
    if positives:
        candidates = qdrant_recommend_by_items(
            positive_skus=positives,
            negative_skus=[],
            limit=body.candidate_limit,
            exclude_skus=exclude_targets,
        )

    if not candidates:
        vec = embed_texts(["diverse catalog best matches"])[0]
        candidates = qdrant_search(vec, limit=body.candidate_limit)

    target_pool: List[Dict[str, str]] = []
    seen_targets: Set[str] = set()
    for r in candidates:
        payload = r.get("payload") or {}
        sku = str(payload.get("sku") or r.get("id"))
        if not sku or sku in exclude_targets or sku in seen_targets:
            continue
        title = str(payload.get("title") or sku)
        target_pool.append({"sku": sku, "title": title})
        seen_targets.add(sku)
    logger.info("prefs_llm: target_pool_size=%d (after exclusions)", len(target_pool))

    if not target_pool:
        return {"customer_id": body.customer_id, "suggestions": []}

    # Source info (for verbs and titles)
    source_payloads = qdrant_payload_for_skus(list({*clicked, *carted, *bought}))
    def title_for(sku: str) -> str:
        p = source_payloads.get(sku) or {}
        return str(p.get("title") or sku)

    sources: List[Tuple[str, str, str]] = []
    for sku in carted:
        sources.append(("added_to_cart", sku, title_for(sku)))
    for sku in clicked:
        sources.append(("clicked", sku, title_for(sku)))
    for sku in bought:
        sources.append(("bought", sku, title_for(sku)))

    if not sources:
        return {"customer_id": body.customer_id, "suggestions": []}

    # Give each source a small set of target options from Qdrant
    OPTIONS_PER_SOURCE = min(8, len(target_pool))
    per_source_options: Dict[str, List[Dict[str, str]]] = {}
    ti = 0
    for (_, src_sku, _) in sources:
        options = []
        tried = 0
        while len(options) < OPTIONS_PER_SOURCE and tried < len(target_pool):
            cand = target_pool[ti]
            ti = (ti + 1) % len(target_pool)
            tried += 1
            if cand["sku"] == src_sku:
                continue
            options.append(cand)
        if options:
            per_source_options[src_sku] = options

    sources = [s for s in sources if s[1] in per_source_options]
    if not sources:
        return {"customer_id": body.customer_id, "suggestions": []}

    llm_items = []
    for (_, src_sku, src_title) in sources:
        llm_items.append({
            "source_sku": src_sku,
            "source_title": src_title,
            "options": per_source_options[src_sku],  # [{sku,title}, ...]
        })

    system = (
        "You are a concise e-commerce copywriter.\n"
        "TASK: For each source item, select ONE target from the provided 'options' and write a short suggestion FRAGMENT about the TARGET only.\n"
        "CONSTRAINTS:\n"
        " - Choose the target strictly from the provided 'options' for that source.\n"
        " - Do NOT mention what the user did (no 'viewed', 'added', 'bought', etc.).\n"
        " - Use ONLY the provided titles; do NOT invent products.\n"
        " - Keep each fragment under 100 characters.\n"
        " - Tone: friendly and helpful.\n"
        " - OUTPUT: JSON array only, no extra text, no code fences.\n"
        '   Each object must be: {"source_sku": string, "target_sku": string, "fragment": string}.\n'
        " - Avoid recommending the same target twice if possible."
    )
    user = (
        "Pick exactly one target for each item and return fragments:\n"
        + json.dumps(llm_items, ensure_ascii=False)
        + "\n\nReturn ONLY a JSON array like:\n"
        + '[{"source_sku":"...","target_sku":"...","fragment":"..."}]'
    )

    def action_verb(a: str) -> str:
        return "viewed" if a == "clicked" else ("added to cart" if a == "added_to_cart" else "bought")

    source_meta: Dict[str, Dict[str, str]] = {s[1]: {"action": s[0], "title": s[2]} for s in sources}
    allowed_targets_per_source: Dict[str, Set[str]] = {
        src: {opt["sku"] for opt in opts} for src, opts in per_source_options.items()
    }
    target_title_map: Dict[str, str] = {t["sku"]: t["title"] for t in target_pool}

    # Human-friendly CTA templates (rotated)
    CTAS = [
        "take a look at",
        "see more about",
        "check out",
        "have a look at",
        "discover",
        "view details for",
    ]

    try:
        out = call_llm(system, user)
        out_clean = out.strip()
        out_clean = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", out_clean)
        data = json.loads(out_clean)
        if not isinstance(data, list):
            raise ValueError("LLM did not return a JSON array")

        used_targets: Set[str] = set()
        suggestions: List[Dict[str, str]] = []
        for idx, obj in enumerate(data):
            if len(suggestions) >= min(body.top_k, MAX_RESULTS):
                break
            if not isinstance(obj, dict):
                continue
            src = str(obj.get("source_sku", "")).strip()
            tgt = str(obj.get("target_sku", "")).strip()
            # Fragment is ignored for final wording; we build deterministic CTA
            if not src or not tgt:
                continue
            if src not in source_meta:
                continue
            if tgt not in allowed_targets_per_source.get(src, set()):
                continue
            if tgt in used_targets:
                continue
            meta = source_meta[src]
            verb = action_verb(meta["action"])
            src_title = meta["title"]
            tgt_title = target_title_map.get(tgt, tgt)
            cta = CTAS[idx % len(CTAS)]
            text = f'You {verb} “{src_title}” — {cta} “{tgt_title}”.'
            suggestions.append({"text": text, "source_sku": src, "target_sku": tgt})
            used_targets.add(tgt)

        logger.info("prefs_llm: suggestions_returned=%d", len(suggestions))
        return {"customer_id": body.customer_id, "suggestions": suggestions}
    except Exception as e:
        logger.error("prefs_llm: LLM failure -> fallback using first options. Error=%s", e)
        logger.debug("prefs_llm traceback:\n%s", traceback.format_exc())
        suggestions: List[Dict[str, str]] = []
        used_targets: Set[str] = set()
        for idx, (action, src_sku, src_title) in enumerate(sources):
            if len(suggestions) >= min(body.top_k, MAX_RESULTS):
                break
            for opt in per_source_options.get(src_sku, []):
                if opt["sku"] != src_sku and opt["sku"] not in used_targets:
                    verb = action_verb(action)
                    cta = CTAS[idx % len(CTAS)]
                    tgt_title = opt["title"]
                    text = f'You {verb} “{src_title}” — {cta} “{tgt_title}”.'
                    suggestions.append({"text": text, "source_sku": src_sku, "target_sku": opt["sku"]})
                    used_targets.add(opt["sku"])
                    break
        return {"customer_id": body.customer_id, "suggestions": suggestions}

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
    logger.info("prefs_points: fallback_candidates=%d", len(candidates if candidates else []))

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
            "title": payload.get("title", ""),   # include title for downstream return
        })

    logger.info("prefs_points: scored=%d", len(scored))

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

    return {"recommendations": top} """