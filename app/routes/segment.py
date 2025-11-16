from app.utils.logging import setup_logger
from fastapi import APIRouter, File, HTTPException, UploadFile, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import base64
from app.database import get_async_db
from app.services.background_tasks import BackgroundTaskService
from app.config import settings

logger = setup_logger(__name__)
router = APIRouter(prefix="/segment", tags=["segmentation"])


@router.post("/")
async def segment(image: UploadFile = File(...), db: AsyncSession = Depends(get_async_db)):
    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an image")
        image_bytes = await image.read()
        logger.info(f"Segmentation request received content_type={image.content_type} size={len(image_bytes)}")
        from app.models.segmentation import SegmentationJob
        job = SegmentationJob(status="pending", model_type=settings.DEFAULT_MODEL, total_masks=0)
        db.add(job)
        await db.commit()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        service = BackgroundTaskService()
        task_id = await service.submit_single_job(
            image_b64=image_b64,
            model_type=settings.DEFAULT_MODEL,
            parameters={},
            job_id=job.id,
        )
        return {
            "success": True,
            "message": "Segmentation started",
            "job_id": job.id,
            "task_id": task_id,
            "status_url": f"/jobs/{job.id}",
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Segmentation failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Segmentation failed: {str(e)}")
