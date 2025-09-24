import json
import logging
from typing import Dict, List, Any, Set, Tuple
from fastapi import HTTPException
from config import MAX_RESULTS
from embeddings import embed_texts
from qdrant import (
    qdrant_recommend_by_items, qdrant_search, qdrant_payload_for_skus
)
from llm_client import call_llm, strip_code_fences

logger = logging.getLogger("recommender")

CTAS = [
    "take a look at",
    "see more about",
    "check out",
    "have a look at",
    "discover",
    "view details for",
]

def action_verb(a: str) -> str:
    return "viewed" if a == "clicked" else ("added to cart" if a == "added_to_cart" else "bought")

def build_suggestions(body) -> Dict[str, Any]:
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

    # Source info
    source_payloads = qdrant_payload_for_skus(list({*clicked, *carted, *bought}))
    def title_for(sku: str) -> str:
        p = source_payloads.get(sku) or {}
        return str(p.get("title") or sku)

    sources: List[Tuple[str, str, str]] = []
    for sku in carted:  sources.append(("added_to_cart", sku, title_for(sku)))
    for sku in clicked: sources.append(("clicked", sku, title_for(sku)))
    for sku in bought:  sources.append(("bought", sku, title_for(sku)))
    if not sources:
        return {"customer_id": body.customer_id, "suggestions": []}

    # options per source
    OPTIONS_PER_SOURCE = min(8, len(target_pool))
    per_source_options: Dict[str, List[Dict[str, str]]] = {}
    ti = 0
    for (_, src_sku, _) in sources:
        options = []
        tried = 0
        while len(options) < OPTIONS_PER_SOURCE and tried < len(target_pool):
            cand = target_pool[ti]; ti = (ti + 1) % len(target_pool); tried += 1
            if cand["sku"] == src_sku:
                continue
            options.append(cand)
        if options:
            per_source_options[src_sku] = options

    sources = [s for s in sources if s[1] in per_source_options]
    if not sources:
        return {"customer_id": body.customer_id, "suggestions": []}

    llm_items = [{"source_sku": s[1], "source_title": s[2], "options": per_source_options[s[1]]} for s in sources]

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
    user = "Pick exactly one target for each item and return fragments:\n" + json.dumps(llm_items, ensure_ascii=False)

    # Build maps for validation and final wording
    source_meta: Dict[str, Dict[str, str]] = {s[1]: {"action": s[0], "title": s[2]} for s in sources}
    allowed_targets_per_source: Dict[str, Set[str]] = {
        src: {opt["sku"] for opt in opts} for src, opts in per_source_options.items()
    }
    target_title_map: Dict[str, str] = {t["sku"]: t["title"] for t in target_pool}

    try:
        out = call_llm(system, user)
        data = json.loads(strip_code_fences(out))
        if not isinstance(data, list):
            raise HTTPException(500, "LLM did not return JSON array")
        used_targets: Set[str] = set()
        suggestions: List[Dict[str, str]] = []
        for idx, obj in enumerate(data):
            if len(suggestions) >= min(body.top_k, MAX_RESULTS):
                break
            if not isinstance(obj, dict):
                continue
            src = str(obj.get("source_sku", "")).strip()
            tgt = str(obj.get("target_sku", "")).strip()
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
        return {"customer_id": body.customer_id, "suggestions": suggestions}
    except Exception as e:
        logger.error("prefs_llm: LLM failure -> fallback. Error=%s", e)
        # Cheap deterministic fallback
        suggestions: List[Dict[str, str]] = []
        used: Set[str] = set()
        idx = 0
        for (action, src, src_title) in sources:
            for opt in per_source_options.get(src, []):
                if opt["sku"] != src and opt["sku"] not in used:
                    cta = CTAS[idx % len(CTAS)]
                    text = f'You {action_verb(action)} “{src_title}” — {cta} “{opt["title"]}”.'
                    suggestions.append({"text": text, "source_sku": src, "target_sku": opt["sku"]})
                    used.add(opt["sku"]); idx += 1
                    if len(suggestions) >= min(body.top_k, MAX_RESULTS):
                        break
            if len(suggestions) >= min(body.top_k, MAX_RESULTS):
                break
        return {"customer_id": body.customer_id, "suggestions": suggestions}
