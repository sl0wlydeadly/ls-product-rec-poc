from typing import Any, Dict, List, Set, Tuple

CLICK_W = 0.6
CART_W  = 0.8
BOUGHT_W = 0.0
TAG_W   = 0.4   # multiplied by Jaccard(tag_candidate, tag_signals)

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

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
