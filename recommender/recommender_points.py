from typing import Any, Dict, List, Set
import logging
from config import MAX_RESULTS, SCORE_THRESHOLD
from embeddings import embed_texts
from qdrant import (
    qdrant_recommend_by_items, qdrant_payload_for_skus, qdrant_search
)
from scoring import score_candidate_unified, jaccard

logger = logging.getLogger("recommender")

def build_signal_tags(clicked: List[str], carted: List[str], bought: List[str]) -> Set[str]:
    payloads = qdrant_payload_for_skus(list({*clicked, *carted, *bought}))
    tags: Set[str] = set()
    for p in payloads.values():
        for t in p.get("tags", []) or []:
            tags.add(str(t))
    logger.info("build_signal_tags: distinct_tags=%d", len(tags))
    return tags

def recommend_points(body) -> Dict[str, Any]:
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

    candidates: List[Dict[str, Any]] = []
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
            "title": payload.get("title", ""),
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
