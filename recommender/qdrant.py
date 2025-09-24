import uuid
import requests
import logging
from typing import List, Dict, Any, Optional, Set
from fastapi import HTTPException
from config import QDRANT, QCOLL

logger = logging.getLogger("recommender")

def ensure_collection(vec_size: int):
    info = requests.get(f"{QDRANT}/collections/{QCOLL}", timeout=10)
    if info.status_code != 200:
        logger.info("index_products: creating collection size=%d distance=Cosine", vec_size)
        create = requests.put(
            f"{QDRANT}/collections/{QCOLL}",
            json={"vectors": {"size": vec_size, "distance": "Cosine"}},
            timeout=30,
        )
        if create.status_code not in (200, 201):
            logger.error("index_products: collection create failed: %s", create.text[:400])
            raise HTTPException(create.status_code, f"Qdrant create error: {create.text}")

def upsert_points(points: List[Dict[str, Any]]):
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
    ids: List[str] = []
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

def make_points(items: List[Dict[str, Any]], vectors: List[List[float]]) -> List[Dict[str, Any]]:
    points = []
    for product, vec in zip(items, vectors):
        points.append(
            {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": {
                    "sku": product["id"],
                    "title": product["title"],
                    "description": product["description"],
                    "tags": product["tags"],
                },
            }
        )
    return points
