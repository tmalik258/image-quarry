from app.utils.logging import setup_logger
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional
from pathlib import Path
from app.services.storage.google_drive import GoogleDriveService
from app.config import settings

logger = setup_logger(__name__)
router = APIRouter(prefix="/storage", tags=["storage"])

@router.get("/drive/status")
async def drive_status():
    try:
        service = GoogleDriveService()
        return service.status()
    except Exception as e:
        logger.error(f"Drive status failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/drive/upload")
async def drive_upload(
    file: UploadFile = File(...),
    folder_id: Optional[str] = Form(default=None),
):
    try:
        if not settings.GOOGLE_DRIVE_ENABLED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google Drive disabled")
        tmp_dir = Path(settings.UPLOAD_DIR)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / file.filename
        content = await file.read()
        tmp_path.write_bytes(content)
        service = GoogleDriveService()
        fid, web = service.upload_file(str(tmp_path), folder_id=folder_id)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {"file_id": fid, "web_view": web}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drive upload failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

