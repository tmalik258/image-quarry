import importlib
import os
from typing import Optional
from app.utils.logging import setup_logger
from app.config import settings
from app.services.model_manager import ModelManager
from app.services.model_manager import model_manager

logger = setup_logger(__name__)


async def ensure_model_ready(model_type: Optional[str] = None) -> None:
    model_type = model_type or settings.DEFAULT_MODEL
    try:
        importlib.import_module("segment_anything")
    except Exception as e:
        logger.error(f"segment_anything import failed: {str(e)}")
        raise RuntimeError(
            "segment_anything is not installed. Install via requirements using GitHub VCS URL."
        )

    try:
        await model_manager.download_model(model_type)
        predictor = await model_manager.load_model(model_type)
        info = await model_manager.get_model_info(model_type)
        model_path = info.get("file_path")
        if not model_path or not os.path.exists(model_path):
            logger.warning("Model weights not found after bootstrap")
            return
        if os.path.getsize(model_path) <= 0:
            logger.warning("Model weights file is empty after bootstrap")
            return
        logger.info(f"Model bootstrap complete: {model_type}, loaded=true path={model_path}")
    except Exception as e:
        logger.error(f"Model bootstrap failed: {str(e)}")
        return
