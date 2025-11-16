from typing import List, Dict, Any, Tuple


def accuracy(preds: List[str], gts: List[str]) -> float:
    if not preds or not gts or len(preds) != len(gts):
        return 0.0
    c = sum(1 for p, g in zip(preds, gts) if p == g)
    return c / len(preds)


def topk_accuracy(results: List[Dict[str, Any]], gts: List[str], k: int = 3) -> float:
    if not results or not gts or len(results) != len(gts):
        return 0.0
    c = 0
    for r, g in zip(results, gts):
        labels = [x["label"] for x in r.get("labels", [])[:k]]
        if g in labels:
            c += 1
    return c / len(results)


def confidence_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    vals = [float(r.get("max_confidence", 0.0)) for r in results]
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}