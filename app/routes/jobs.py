from app.utils.logging import setup_logger
from fastapi import APIRouter, Depends, HTTPException, Query, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional
import os
import glob
from datetime import datetime, timedelta

from app.database import get_async_db
from app.api_schema.segmentation import JobStatusResponse
from app.models.segmentation import SegmentationJob, BatchJob, SegmentationMask
from app.utils.exceptions import JobNotFoundError

logger = setup_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/", response_model=List[JobStatusResponse])
async def list_jobs(
    status_filter: Optional[str] = Query(None, description="Filter by job status"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    db: AsyncSession = Depends(get_async_db),
):
    """List segmentation jobs with optional filtering."""
    
    try:
        query = select(SegmentationJob)
        
        if status_filter:
            query = query.where(SegmentationJob.status == status_filter)
        
        if model_type:
            query = query.where(SegmentationJob.model_type == model_type)
        
        query = query.order_by(SegmentationJob.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        result = await db.execute(query)
        jobs = result.scalars().all()
        
        return [
            JobStatusResponse(
                job_id=str(job.id),
                status=job.status,
                model_type=job.model_type,
                total_masks=job.total_masks,
                created_at=job.created_at,
                updated_at=job.updated_at,
                error_message=job.error_message,
                batch_job_id=str(job.batch_job_id) if job.batch_job_id else None,
            )
            for job in jobs
        ]
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}",
        )


@router.get("/batch", response_model=List[dict])
async def list_batch_jobs(
    status_filter: Optional[str] = Query(None, description="Filter by batch job status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of batch jobs to return"),
    offset: int = Query(0, ge=0, description="Number of batch jobs to skip"),
    db: AsyncSession = Depends(get_async_db),
):
    """List batch segmentation jobs with optional filtering."""
    
    try:
        query = select(BatchJob)
        
        if status_filter:
            query = query.where(BatchJob.status == status_filter)
        
        query = query.order_by(BatchJob.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        result = await db.execute(query)
        batch_jobs = result.scalars().all()
        
        return [
            {
                "batch_job_id": str(job.id),
                "status": job.status,
                "total_images": job.total_images,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "error_message": job.error_message,
                "task_id": job.task_id,
            }
            for job in batch_jobs
        ]
        
    except Exception as e:
        logger.error(f"Failed to list batch jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list batch jobs: {str(e)}",
        )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_async_db),
):
    """Get detailed status of a specific job."""
    
    try:
        result = await db.execute(
            select(SegmentationJob).where(SegmentationJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        
        if not job:
            raise JobNotFoundError(job_id)
        
        return JobStatusResponse(
            job_id=str(job.id),
            status=job.status,
            model_type=job.model_type,
            total_masks=job.total_masks,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error_message=job.error_message,
            batch_job_id=str(job.batch_job_id) if job.batch_job_id else None,
        )
        
    except JobNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}",
        )


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    db: AsyncSession = Depends(get_async_db),
):
    """Delete a segmentation job and its associated data."""
    
    try:
        result = await db.execute(
            select(SegmentationJob).where(SegmentationJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        
        if not job:
            raise JobNotFoundError(job_id)
        
        # Delete associated masks first (due to foreign key constraint)
        from sqlalchemy import delete
        await db.execute(
            delete(SegmentationMask).where(SegmentationMask.job_id == job_id)
        )
        
        # Delete the job
        await db.delete(job)
        await db.commit()
        
        return {"message": f"Job {job_id} deleted successfully"}
        
    except JobNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete job: {str(e)}",
        )


@router.delete("/batch/{batch_job_id}")
async def delete_batch_job(
    batch_job_id: str,
    db: AsyncSession = Depends(get_async_db),
):
    """Delete a batch job and all its associated individual jobs."""
    
    try:
        result = await db.execute(
            select(BatchJob).where(BatchJob.id == batch_job_id)
        )
        batch_job = result.scalar_one_or_none()
        
        if not batch_job:
            raise JobNotFoundError(batch_job_id)
        
        # Get all individual jobs in this batch
        result = await db.execute(
            select(SegmentationJob).where(SegmentationJob.batch_job_id == batch_job_id)
        )
        individual_jobs = result.scalars().all()
        
        # Delete masks for all individual jobs
        from sqlalchemy import delete
        job_ids = [job.id for job in individual_jobs]
        if job_ids:
            await db.execute(
                delete(SegmentationMask).where(SegmentationMask.job_id.in_(job_ids))
            )
        
        # Delete all individual jobs
        for job in individual_jobs:
            await db.delete(job)
        
        # Delete the batch job
        await db.delete(batch_job)
        await db.commit()
        
        return {"message": f"Batch job {batch_job_id} and all associated jobs deleted successfully"}
        
    except JobNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to delete batch job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete batch job: {str(e)}",
        )


@router.post("/cleanup")
async def cleanup_old_jobs(
    days_old: int = Query(7, ge=1, description="Delete jobs older than this many days"),
    db: AsyncSession = Depends(get_async_db),
):
    """Clean up old completed jobs."""
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Get jobs to delete
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
            from sqlalchemy import delete
            await db.execute(
                delete(SegmentationMask).where(SegmentationMask.job_id.in_(job_ids))
            )
            
            # Delete jobs
            await db.execute(
                delete(SegmentationJob).where(SegmentationJob.id.in_(job_ids))
            )
        
        await db.commit()
        
        return {
            "message": f"Cleaned up {deleted_count} old jobs",
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup old jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup old jobs: {str(e)}",
        )


@router.get("/{job_id}/masks")
async def list_job_masks(job_id: str, db: AsyncSession = Depends(get_async_db)):
    try:
        result = await db.execute(select(SegmentationJob).where(SegmentationJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            raise JobNotFoundError(job_id)
        result = await db.execute(
            select(SegmentationMask)
            .where(SegmentationMask.job_id == job_id)
            .order_by(SegmentationMask.mask_index.asc())
        )
        masks = result.scalars().all()
        return {
            "job_id": job_id,
            "total": len(masks),
            "items": [
                {
                    "mask_index": m.mask_index,
                    "area": m.area,
                    "confidence": m.confidence,
                    "bbox": m.bbox,
                }
                for m in masks
            ],
        }
    except JobNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to list job masks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list job masks: {str(e)}",
        )


@router.get("/{job_id}/masks/{mask_index}")
async def get_job_mask_image(job_id: str, mask_index: int, db: AsyncSession = Depends(get_async_db)):
    try:
        result = await db.execute(
            select(SegmentationMask).where(
                (SegmentationMask.job_id == job_id) & (SegmentationMask.mask_index == mask_index)
            )
        )
        m = result.scalar_one_or_none()
        if not m:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mask not found")
        return Response(content=m.mask_data, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get mask image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get mask image: {str(e)}",
        )


@router.get("/{job_id}/files")
async def list_job_files(job_id: str):
    try:
        from app.config import settings
        base = settings.RESULTS_DIR
        job_dir = os.path.join(base, job_id)
        mask_dir = os.path.join(job_dir, "masks")
        obj_dir = os.path.join(job_dir, "objects")
        masks = sorted(glob.glob(os.path.join(mask_dir, "mask_*.png")))
        objects = sorted(glob.glob(os.path.join(obj_dir, "object_*.png")))
        overlay = os.path.join(job_dir, "overlay.png")
        labels = os.path.join(job_dir, "labels.jsonl")
        source_enhanced = os.path.join(job_dir, "source_enhanced.png")
        quality = os.path.join(job_dir, "quality.json")
        diff = os.path.join(job_dir, "diff.png")
        return {
            "job_id": job_id,
            "directory": job_dir,
            "mask_dir": mask_dir,
            "object_dir": obj_dir,
            "masks": masks,
            "objects": objects,
            "overlay": overlay if os.path.exists(overlay) else None,
            "labels": labels if os.path.exists(labels) else None,
            "source_enhanced": source_enhanced if os.path.exists(source_enhanced) else None,
            "quality": quality if os.path.exists(quality) else None,
            "diff": diff if os.path.exists(diff) else None,
        }
    except Exception as e:
        logger.error(f"Failed to list job files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list job files: {str(e)}",
        )


@router.get("/{job_id}/comparison")
async def get_job_comparison_image(job_id: str, db: AsyncSession = Depends(get_async_db)):
    try:
        from app.config import settings
        from PIL import Image
        import io
        import base64
        import numpy as np
        base = settings.RESULTS_DIR
        job_dir = os.path.join(base, job_id)
        source_path = os.path.join(job_dir, "source.png")
        overlay_path = os.path.join(job_dir, "overlay.png")
        result = await db.execute(select(SegmentationJob).where(SegmentationJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            raise JobNotFoundError(job_id)
        if not os.path.exists(overlay_path) and os.path.exists(source_path):
            result = await db.execute(
                select(SegmentationMask)
                .where(SegmentationMask.job_id == job_id)
                .order_by(SegmentationMask.mask_index.asc())
            )
            masks = result.scalars().all()
            if masks:
                try:
                    from app.services.image_processor import ImageProcessor
                    processor = ImageProcessor()
                    img_rgb = np.array(Image.open(source_path).convert('RGB'))
                    mask_dicts = [
                        {
                            "segmentation": base64.b64encode(m.mask_data).decode("utf-8"),
                            "area": m.area or 0,
                            "bbox": m.bbox,
                            "predicted_iou": m.confidence,
                        }
                        for m in masks
                    ]
                    overlay_b64 = processor.create_visualization(img_rgb, mask_dicts)
                    with open(overlay_path, "wb") as f:
                        f.write(base64.b64decode(overlay_b64))
                except Exception as e:
                    logger.error(f"Failed to generate overlay for comparison job_id={job_id} error={str(e)}")
        if not os.path.exists(source_path) or not os.path.exists(overlay_path):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comparison images not available")
        left = Image.open(source_path).convert('RGB')
        right = Image.open(overlay_path).convert('RGB')
        if right.size != left.size:
            right = right.resize(left.size)
        combined = Image.new('RGB', (left.width + right.width, left.height))
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))
        buf = io.BytesIO()
        combined.save(buf, format='PNG')
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except JobNotFoundError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate comparison image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate comparison image: {str(e)}",
        )