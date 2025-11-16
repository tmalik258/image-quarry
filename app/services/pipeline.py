import asyncio
import base64
import io
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

from app.utils.logging import setup_logger
from app.config import settings
from app.services.image_processor import ImageProcessor
from app.services.object_extractor import ObjectExtractor
from app.services.model_manager import model_manager

logger = setup_logger(__name__)


class PipelineResult:
    def __init__(self, job_id: str, output_dir: str, files: Dict[str, Any], objects_count: int, parameters: Dict[str, Any]):
        self.job_id = job_id
        self.output_dir = output_dir
        self.files = files
        self.objects_count = objects_count
        self.parameters = parameters


async def process_image_pipeline(
    image_bytes: bytes,
    content_type: str,
    model_type: Optional[str] = None,
    min_pixels: int = 5000,
) -> PipelineResult:
    processor = ImageProcessor()
    ok, msg = processor.validate_image(image_bytes, content_type)
    if not ok:
        raise ValueError(msg)

    job_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    output_dir = os.path.join(settings.RESULTS_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    import time
    t_start = time.perf_counter()
    image_rgba, image_rgb = await asyncio.to_thread(processor.preprocess_image_pair, image_bytes)
    try:
        import psutil
        rss_mb = float(psutil.Process(os.getpid()).memory_info().rss) / (1024**2)
        logger.info(f"Resource snapshot location=Pipeline.process_image_pipeline stage=preprocess rss_mb={rss_mb:.2f} job_id={job_id}")
    except Exception:
        pass

    try:
        t_load0 = time.perf_counter()
        predictor = await model_manager.load_model(model_type or settings.DEFAULT_MODEL)
    except Exception:
        # Retry once after CUDA cache clear
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        predictor = await model_manager.load_model(model_type or settings.DEFAULT_MODEL)
    try:
        import torch
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024**2)
            mem_res = torch.cuda.memory_reserved() / (1024**2)
            logger.info(f"GPU memory after load alloc_mb={mem_alloc:.2f} reserved_mb={mem_res:.2f} job_id={job_id}")
    except Exception:
        pass
    logger.info(f"Model ready in {time.perf_counter() - t_load0:.2f}s job_id={job_id}")

    auto_params = {
        "points_per_side": 64,
        "pred_iou_thresh": 0.90,
        "stability_score_thresh": 0.92,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 1000,
    }

    t_seg0 = time.perf_counter()
    masks = await processor.segment_image(
        image_data=image_bytes,
        model_type=model_type or settings.DEFAULT_MODEL,
        auto_params=auto_params,
        predictor=predictor,
        job_id=job_id,
    )
    logger.info(f"Segmentation stage took {time.perf_counter() - t_seg0:.2f}s, masks={len(masks)} job_id={job_id}")

    # Persist masks to files
    mask_paths: List[str] = []
    for i, m in enumerate(masks, start=1):
        try:
            b64 = m.get("segmentation") or m.get("mask")
            if not b64:
                continue
            data = base64.b64decode(b64)
            path = os.path.join(mask_dir, f"mask_{i}.png")
            with open(path, "wb") as f:
                f.write(data)
            if os.path.getsize(path) > 0:
                mask_paths.append(path)
            else:
                os.remove(path)
        except Exception as e:
            logger.error(f"Failed to save mask {i}: {str(e)}")

    extractor = ObjectExtractor(min_pixels=min_pixels)
    decoded = []
    for m in masks:
        try:
            arr = processor._base64_to_mask(m["segmentation"], image_rgb.shape[:2])
            try:
                arr = processor.refine_mask(arr)
            except Exception:
                pass
            decoded.append({"segmentation": arr})
        except Exception:
            continue

    t_extract0 = time.perf_counter()
    objects = await asyncio.to_thread(extractor.extract, image_rgb, decoded)
    object_dir = os.path.join(output_dir, "objects")
    os.makedirs(object_dir, exist_ok=True)
    object_paths = await asyncio.to_thread(extractor.save_objects, objects, object_dir)
    logger.info(f"Object extraction took {time.perf_counter() - t_extract0:.2f}s, objects={len(object_paths)}")

    # Visualization overlay
    t_vis0 = time.perf_counter()
    overlay_b64 = processor.create_visualization(image_rgba, masks)
    overlay_path = os.path.join(output_dir, "overlay.png")
    with open(overlay_path, "wb") as f:
        f.write(base64.b64decode(overlay_b64))
    logger.info(f"Visualization took {time.perf_counter() - t_vis0:.2f}s")

    # Quality enhancement and QA artifacts for pipeline mode
    try:
        from app.services.quality import QualityEnhancer
        enh = QualityEnhancer(settings)
        enhanced = enh.enhance(image_bytes, content_type=content_type, job_dir=output_dir)
    except Exception as e:
        logger.error(f"Quality enhancement (pipeline) failed job_id={job_id} error={str(e)}")

    # Metadata
    metadata = {
        "job_id": job_id,
        "model_type": model_type or settings.DEFAULT_MODEL,
        "counts": {
            "masks": len(mask_paths),
            "objects": len(object_paths),
        },
        "parameters": auto_params,
        "timestamps": {"started": datetime.utcnow().isoformat()},
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    files = {
        "overlay": overlay_path if os.path.getsize(overlay_path) > 0 else None,
        "masks": mask_paths,
        "objects": object_paths,
        "metadata": metadata_path,
        "source_enhanced": os.path.join(output_dir, "source_enhanced.png") if os.path.exists(os.path.join(output_dir, "source_enhanced.png")) else None,
        "quality": os.path.join(output_dir, "quality.json") if os.path.exists(os.path.join(output_dir, "quality.json")) else None,
        "diff": os.path.join(output_dir, "diff.png") if os.path.exists(os.path.join(output_dir, "diff.png")) else None,
    }

    logger.info(f"Pipeline completed in {time.perf_counter() - t_start:.2f}s")
    return PipelineResult(
        job_id=job_id,
        output_dir=output_dir,
        files=files,
        objects_count=len(object_paths),
        parameters=auto_params,
    )