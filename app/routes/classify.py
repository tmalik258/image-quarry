from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from typing import Optional, List, Dict, Any
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_async_db
from app.utils.logging import setup_logger
from app.services.clip_classifier import ClipClassifier
from app.models.segmentation import SegmentationJob, SegmentationMask


logger = setup_logger(__name__)
router = APIRouter(prefix="/classify", tags=["classify"])
_clip: Optional[ClipClassifier] = None


def _get_clip() -> ClipClassifier:
    global _clip
    if _clip is None:
        _clip = ClipClassifier()
    return _clip


@router.post("/image")
async def classify_image(
    image: UploadFile = File(...),
    min_confidence: float = Form(0.25),
    top_k: int = Form(3),
):
    try:
        data = await image.read()
        clip = _get_clip()
        img = clip.load_image(data)
        img_feats = clip.encode_images([img])
        labels = clip.build_concepts()
        text_feats = clip.encode_texts(labels)
        res = clip.classify(img_feats, text_feats, labels, top_k=top_k)
        filtered = [r for r in res if r["max_confidence"] >= min_confidence]
        return {"items": filtered}
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/job/{job_id}")
async def classify_job_assets(
    job_id: str,
    min_confidence: float = Form(0.25),
    top_k: int = Form(3),
    export: bool = Form(True),
    remove_below_threshold: bool = Form(False),
    group: bool = Form(False),
    db: AsyncSession = Depends(get_async_db),
):
    try:
        clip = _get_clip()
        from app.config import settings
        base = settings.RESULTS_DIR
        job_dir = os.path.join(base, job_id)
        source_path = os.path.join(job_dir, "source.png")
        overlay_path = os.path.join(job_dir, "overlay.png")
        mask_dir = os.path.join(job_dir, "masks")
        obj_dir = os.path.join(job_dir, "objects")
        paths: List[str] = []
        images: List[Any] = []
        if os.path.exists(source_path):
            paths.append(source_path)
            images.append(clip.load_image(open(source_path, "rb").read()))
        if os.path.exists(overlay_path):
            paths.append(overlay_path)
            images.append(clip.load_image(open(overlay_path, "rb").read()))
        if os.path.isdir(mask_dir):
            for name in sorted(os.listdir(mask_dir)):
                p = os.path.join(mask_dir, name)
                if os.path.isfile(p):
                    paths.append(p)
                    images.append(clip.load_image(open(p, "rb").read()))
        if os.path.isdir(obj_dir):
            for name in sorted(os.listdir(obj_dir)):
                p = os.path.join(obj_dir, name)
                if os.path.isfile(p):
                    paths.append(p)
                    images.append(clip.load_image(open(p, "rb").read()))
        if not images:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No assets found")
        img_feats = clip.encode_images(images)
        labels = clip.build_concepts()
        text_feats = clip.encode_texts(labels)
        res = clip.classify(img_feats, text_feats, labels, top_k=top_k)
        if export:
            out_path = os.path.join(job_dir, "labels.jsonl")
            clip.export_jsonl(res, paths, out_path)
        if remove_below_threshold:
            keep_idx = [i for i, r in enumerate(res) if r["max_confidence"] >= min_confidence]
            for i, p in enumerate(paths):
                if i not in keep_idx and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            removed_masks: List[int] = []
            for i, p in enumerate(paths):
                if i not in keep_idx and os.path.basename(p).startswith("mask_") and p.endswith(".png"):
                    try:
                        name = os.path.basename(p)
                        idx = int(name.split("_")[1].split(".")[0]) - 1
                        removed_masks.append(idx)
                    except Exception:
                        pass
            if removed_masks:
                from sqlalchemy import delete
                await db.execute(
                    delete(SegmentationMask).where(
                        (SegmentationMask.job_id == job_id) & (SegmentationMask.mask_index.in_(removed_masks))
                    )
                )
            await db.commit()
        payload = {"count": len(res), "labels_path": os.path.join(job_dir, "labels.jsonl"), "items": res}
        if group:
            payload["groups"] = _get_clip().group_by_top1(res)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/batch")
async def classify_batch(
    images: List[UploadFile] = File(...),
    min_confidence: float = Form(0.25),
    top_k: int = Form(3),
):
    try:
        clip = _get_clip()
        pil_list = []
        for uf in images:
            data = await uf.read()
            pil_list.append(clip.load_image(data))
        feats = clip.encode_images(pil_list)
        labels = clip.build_concepts()
        text_feats = clip.encode_texts(labels)
        res = clip.classify(feats, text_feats, labels, top_k=top_k)
        items = [r for r in res if r["max_confidence"] >= min_confidence]
        return {"count": len(items), "items": items}
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/metrics")
async def classification_metrics(results: List[Dict[str, Any]], ground_truth: List[str], k: int = 3):
    try:
        from app.services.metrics import accuracy, topk_accuracy, confidence_stats
        preds = [r.get("top1") for r in results]
        return {
            "accuracy": accuracy(preds, ground_truth),
            "topk": topk_accuracy(results, ground_truth, k=k),
            "confidence": confidence_stats(results),
        }
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))