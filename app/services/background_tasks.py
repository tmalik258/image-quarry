import asyncio
import base64
import uuid
from typing import Dict, Any, List
from app.utils.logging import setup_logger
from celery import Celery

from app.config import settings
from app.services.image_processor import ImageProcessor
from app.services.model_manager import ModelManager
from app.services.model_manager import model_manager
from app.services.object_extractor import ObjectExtractor
from app.services.storage.google_drive import GoogleDriveService
from app.models.segmentation import SegmentationJob, SegmentationMask, BatchJob
from app.utils.exceptions import ImageValidationError, ModelError

logger = setup_logger(__name__)

# Initialize Celery
celery_app = Celery(
    "image_quarry",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)


class BackgroundTaskService:
    """Service for managing background tasks."""
    
    def __init__(self):
        self.celery = celery_app
    
    async def submit_batch_job(
        self,
        batch_job_id: str,
        images: List[Dict[str, Any]],
        model_type: str,
        parameters: Dict[str, Any] = None,
    ) -> str:
        """Submit a batch segmentation job."""
        
        task_id = str(uuid.uuid4())
        
        # Submit task to Celery
        task = process_batch_job.delay(
            batch_job_id=batch_job_id,
            images=images,
            model_type=model_type,
            parameters=parameters or {},
        )
        
        logger.info(
            f"Batch job submitted batch_job_id={batch_job_id} task_id={task.id} total_images={len(images)} model_type={model_type}"
        )
        
        return task.id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a background task."""
        
        task = self.celery.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "status": task.status,
            "result": task.result if task.ready() else None,
            "error": str(task.result) if task.failed() else None,
        }

    async def submit_single_job(
        self,
        image_b64: str,
        model_type: str,
        parameters: Dict[str, Any] = None,
        job_id: str = None,
    ) -> str:
        task = segment_pipeline_task.delay(
            image_b64=image_b64,
            model_type=model_type,
            params=parameters or {},
            job_id=job_id,
        )
        logger.info(
            f"Segmentation job submitted job_id={job_id} task_id={task.id} model_type={model_type}"
        )
        return task.id


@celery_app.task(bind=True, name="process_batch_job")
def process_batch_job(
    self,
    batch_job_id: str,
    images: List[Dict[str, Any]],
    model_type: str,
    parameters: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Process a batch of images for segmentation."""
    
    import asyncio
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.database import AsyncSessionLocal
    
    # Run the async function in a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _process_batch_job_async(
                batch_job_id=batch_job_id,
                images=images,
                model_type=model_type,
                parameters=parameters or {},
                task=self,
            )
        )
        return result
    finally:
        loop.close()


async def _process_batch_job_async(
    batch_job_id: str,
    images: List[Dict[str, Any]],
    model_type: str,
    parameters: Dict[str, Any],
    task,
) -> Dict[str, Any]:
    """Async implementation of batch job processing."""
    
    async with AsyncSessionLocal() as db:
        try:
            # Update batch job status
            from sqlalchemy import select
            result = await db.execute(
                select(BatchJob).where(BatchJob.id == batch_job_id)
            )
            batch_job = result.scalar_one_or_none()
            
            if not batch_job:
                raise ValueError(f"Batch job {batch_job_id} not found")
            
            batch_job.status = "processing"
            await db.commit()
            
            # Load model
            model_loaded = await model_manager.load_model(model_type)
            if not model_loaded:
                raise ModelError(f"Failed to load model {model_type}")
            
            # Process each image
            processor = ImageProcessor()
            completed_count = 0
            failed_count = 0
            
            for i, image_data in enumerate(images):
                try:
                    # Update progress
                    progress = int((i + 1) / len(images) * 100)
                    task.update_state(
                        state="PROGRESS",
                        meta={"current": i + 1, "total": len(images), "progress": progress},
                    )
                    
                    # Decode image data
                    if image_data.get("base64_data"):
                        image_bytes = base64.b64decode(image_data["base64_data"])
                    else:
                        logger.warning("No image data provided for batch item", index=i)
                        failed_count += 1
                        continue
                    
                    # Create individual job
                    job = SegmentationJob(
                        batch_job_id=batch_job_id,
                        status="pending",
                        model_type=model_type,
                        total_masks=0,
                    )
                    db.add(job)
                    await db.flush()
                    
                    # Parse input points/boxes if provided
                    input_points = parameters.get("points")
                    input_boxes = parameters.get("boxes")
                    
                    # Perform segmentation
                    masks = await processor.segment_image(
                        image_data=image_bytes,
                        model_type=model_type,
                        points=input_points,
                        boxes=input_boxes,
                    )
                    
                    # Store masks
                    for j, mask_data in enumerate(masks):
                        mask = SegmentationMask(
                            job_id=job.id,
                            mask_index=j,
                            mask_data=mask_data["mask"],
                            confidence=mask_data.get("confidence", 1.0),
                            bbox=mask_data.get("bbox"),
                            area=mask_data.get("area"),
                        )
                        db.add(mask)
                    
                    # Update job status
                    job.status = "completed"
                    job.total_masks = len(masks)
                    completed_count += 1
                    
                    await db.commit()
                    
                    logger.info(
                        f"Batch job item {i} completed: {len(masks)} masks generated",
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Batch job item {i} failed: {str(e)}",
                    )
                    
                    # Update job status to failed
                    if 'job' in locals():
                        job.status = "failed"
                        job.error_message = str(e)
                        await db.commit()
                    
                    failed_count += 1
            
            # Update batch job status
            batch_job.status = "completed"
            batch_job.completed_at = datetime.utcnow()
            await db.commit()
            
            logger.info(
                "Batch job completed",
                batch_job_id=batch_job_id,
                completed_count=completed_count,
                failed_count=failed_count,
                total_images=len(images),
            )
            
            return {
                "batch_job_id": batch_job_id,
                "status": "completed",
                "completed_count": completed_count,
                "failed_count": failed_count,
                "total_images": len(images),
            }
            
        except Exception as e:
            logger.error(
                f"Batch job {batch_job_id} failed: {str(e)}"
            )
            
            # Update batch job status to failed
            try:
                if batch_job:
                    batch_job.status = "failed"
                    batch_job.error_message = str(e)
                    batch_job.completed_at = datetime.utcnow()
                    await db.commit()
            except Exception as update_error:
                logger.error(
                    f"Failed to update batch job {batch_job_id} status: {str(update_error)}"
                )
            
            raise


@celery_app.task(name="cleanup_old_jobs")
def cleanup_old_jobs_task(days_old: int = 7) -> Dict[str, Any]:
    """Clean up old completed jobs."""
    
    import asyncio
    from datetime import datetime, timedelta
    from sqlalchemy import and_, delete
    from app.database import AsyncSessionLocal
    
    async def cleanup_async():
        async with AsyncSessionLocal() as db:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Get jobs to delete
            from app.models.segmentation import SegmentationJob, SegmentationMask
            
            result = await db.execute(
                select(SegmentationJob).where(
                    and_(
                        SegmentationJob.status.in_(["completed", "failed"]),
                        SegmentationJob.created_at < cutoff_date,
                        SegmentationJob.batch_job_id.is_(None),  # Don't delete batch job components
                    )
                )
            )
            old_jobs = result.scalars().all()
            
            job_ids = [job.id for job in old_jobs]
            deleted_count = len(job_ids)
            
            if job_ids:
                # Delete masks
                await db.execute(
                    delete(SegmentationMask).where(SegmentationMask.job_id.in_(job_ids))
                )
                
                # Delete jobs
                await db.execute(
                    delete(SegmentationJob).where(SegmentationJob.id.in_(job_ids))
                )
            
            await db.commit()
            
            return {
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
            }
    
    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(cleanup_async())
        return result
    finally:
        loop.close()
@celery_app.task(bind=True, name="segment_auto_task")
def segment_auto_task(self, image_b64: str, model_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_segment_auto_async(image_b64, model_type, params or {}))
        return result
    finally:
        loop.close()


async def _segment_auto_async(image_b64: str, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    image_bytes = base64.b64decode(image_b64)
    processor = ImageProcessor()
    masks = await processor.segment_image(
        image_data=image_bytes,
        model_type=model_type,
        auto_params=params,
    )
    return {"masks": masks}


@celery_app.task(bind=True, name="segment_box_task")
def segment_box_task(self, image_b64: str, model_type: str, box: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_segment_box_async(image_b64, model_type, box, params or {}))
        return result
    finally:
        loop.close()


async def _segment_box_async(image_b64: str, model_type: str, box: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    image_bytes = base64.b64decode(image_b64)
    processor = ImageProcessor()
    masks = await processor.segment_image(
        image_data=image_bytes,
        model_type=model_type,
        boxes=[box],
        multimask_output=bool(params.get("multimask_output", True)),
    )
    return {"masks": masks}


@celery_app.task(bind=True, name="extract_objects_task")
def extract_objects_task(self, image_b64: str, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_extract_objects_async(image_b64, model_type, params))
        return result
    finally:
        loop.close()


async def _extract_objects_async(image_b64: str, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    image_bytes = base64.b64decode(image_b64)
    processor = ImageProcessor()
    masks = await processor.segment_image(
        image_data=image_bytes,
        model_type=model_type,
        auto_params={
            "points_per_side": int(params.get("points_per_side", 64)),
            "pred_iou_thresh": float(params.get("pred_iou_thresh", 0.90)),
            "stability_score_thresh": float(params.get("stability_score_thresh", 0.92)),
            "crop_n_layers": int(params.get("crop_n_layers", 1)),
            "crop_n_points_downscale_factor": int(params.get("crop_n_points_downscale_factor", 2)),
            "min_mask_region_area": int(params.get("min_mask_region_area", 1000)),
        },
    )
    import numpy as np
    from PIL import Image
    image_rgb = processor.preprocess_image(image_bytes)
    extractor = ObjectExtractor(min_pixels=int(params.get("min_pixels", 5000)))
    decoded_masks = []
    for m in masks:
        mask_png_b64 = m["segmentation"]
        mask_array = processor._base64_to_mask(mask_png_b64, image_rgb.shape[:2])
        decoded_masks.append({"segmentation": mask_array})
    objects = extractor.extract(image_rgb, decoded_masks)
    save_dir = params.get("save_dir") or settings.RESULTS_DIR
    paths = extractor.save_objects(objects, save_dir)
    drive_files = []
    if bool(params.get("save_to_drive", False)):
        try:
            drive = GoogleDriveService()
            folder_id = params.get("drive_folder_id")
            for p in paths:
                file_id, web_view = drive.upload_file(p, folder_id=folder_id)
                drive_files.append({"file_id": file_id, "web_view": web_view})
        except Exception as e:
            logger.error(f"Drive upload failed: {str(e)}")
    return {"saved": paths, "skipped": len(masks) - len(paths), "drive_files": drive_files}


@celery_app.task(bind=True, name="segment_pipeline_task", autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def segment_pipeline_task(self, image_b64: str, model_type: str, params: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_segment_pipeline_async(self, image_b64, model_type, params, job_id))
        return result
    finally:
        loop.close()


async def _segment_pipeline_async(task, image_b64: str, model_type: str, params: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            from sqlalchemy import select
            result = await db.execute(select(SegmentationJob).where(SegmentationJob.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                job = SegmentationJob(id=job_id, status="pending", model_type=model_type, total_masks=0)
                db.add(job)
                await db.commit()
            job.status = "processing"
            await db.commit()
            await model_manager.load_model(model_type)
            image_bytes = base64.b64decode(image_b64)
            processor = ImageProcessor()
            masks = await processor.segment_image(
                image_data=image_bytes,
                model_type=model_type,
                auto_params={
                    "points_per_side": int(params.get("points_per_side", 64)),
                    "pred_iou_thresh": float(params.get("pred_iou_thresh", 0.90)),
                    "stability_score_thresh": float(params.get("stability_score_thresh", 0.92)),
                    "crop_n_layers": int(params.get("crop_n_layers", 1)),
                    "crop_n_points_downscale_factor": int(params.get("crop_n_points_downscale_factor", 2)),
                    "min_mask_region_area": int(params.get("min_mask_region_area", 1000)),
                },
            )
            import os
            base_dir = params.get("save_dir") or settings.RESULTS_DIR
            save_dir = os.path.join(base_dir, job_id)
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception:
                pass
            mask_dir = os.path.join(save_dir, "masks")
            try:
                os.makedirs(mask_dir, exist_ok=True)
            except Exception:
                pass
            for j, m in enumerate(masks):
                data_b64 = m.get("segmentation") or m.get("mask")
                if not data_b64:
                    continue
                try:
                    import os
                    mask_bytes = base64.b64decode(data_b64)
                    mask_path = os.path.join(mask_dir, f"mask_{j+1}.png")
                    with open(mask_path, "wb") as f:
                        f.write(mask_bytes)
                except Exception as e:
                    logger.error(f"Failed to save mask file index={j+1} dir={save_dir} error={str(e)}")
                db.add(
                    SegmentationMask(
                        job_id=job.id,
                        mask_index=j,
                        mask_data=base64.b64decode(data_b64),
                        confidence=float(m.get("predicted_iou", 1.0)),
                        bbox=m.get("bbox"),
                        area=int(m.get("area")) if m.get("area") is not None else None,
                    )
                )
            try:
                from PIL import Image
                image_rgb = processor.preprocess_image(image_bytes)
                extractor = ObjectExtractor(min_pixels=int(params.get("min_pixels", 5000)))
                decoded_masks = []
                for m in masks:
                    b64 = m.get("segmentation") or m.get("mask")
                    if not b64:
                        continue
                    arr = processor._base64_to_mask(b64, image_rgb.shape[:2])
                    decoded_masks.append({"segmentation": arr})
                objects = extractor.extract(image_rgb, decoded_masks)
                object_dir = os.path.join(save_dir, "objects")
                try:
                    os.makedirs(object_dir, exist_ok=True)
                except Exception:
                    pass
                _ = extractor.save_objects(objects, object_dir)
                # Save source image for later comparison
                try:
                    source_path = os.path.join(save_dir, "source.png")
                    Image.fromarray(image_rgb).save(source_path)
                except Exception as e:
                    logger.error(f"Failed to save source image job_id={job_id} error={str(e)}")
                # Create and save overlay visualization
                try:
                    overlay_b64 = processor.create_visualization(image_rgb, masks)
                    overlay_path = os.path.join(save_dir, "overlay.png")
                    with open(overlay_path, "wb") as f:
                        f.write(base64.b64decode(overlay_b64))
                except Exception as e:
                    logger.error(f"Failed to save overlay visualization job_id={job_id} error={str(e)}")
            except Exception as e:
                logger.error(f"Object extraction failed job_id={job_id} error={str(e)}")
            job.status = "completed"
            job.total_masks = len(masks)
            await db.commit()
            task.update_state(state="PROGRESS", meta={"progress": 1.0, "masks": len(masks)})
            return {"job_id": job.id, "masks": len(masks)}
        except Exception as e:
            try:
                if 'job' in locals() and job:
                    job.status = "failed"
                    job.error_message = str(e)
                    await db.commit()
            except Exception:
                pass
            raise
