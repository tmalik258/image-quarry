import asyncio
import os
import uuid
from pathlib import Path
from typing import Dict, Optional
import aiohttp
import aiofiles
from datetime import datetime
from app.utils.logging import setup_logger
import structlog
from segment_anything import sam_model_registry, SamPredictor
import torch

from app.config import settings

logger = setup_logger(__name__)


class ModelManager:
    """Manages SAM model downloading, caching, and loading."""
    
    MODEL_CONFIG = {
        "vit_h": {
            "url": settings.SAM_MODEL_URLS.get("vit_h"),
            "filename": "sam_vit_h_4b8939.pth",
            "size": "2.4GB",
            "description": "SAM ViT-H (Huge) model"
        },
        "vit_l": {
            "url": settings.SAM_MODEL_URLS.get("vit_l"),
            "filename": "sam_vit_l_0b3195.pth",
            "size": "1.2GB",
            "description": "SAM ViT-L (Large) model"
        },
        "vit_b": {
            "url": settings.SAM_MODEL_URLS.get("vit_b"),
            "filename": "sam_vit_b_01ec64.pth",
            "size": "375MB",
            "description": "SAM ViT-B (Base) model"
        }
    }
    MIN_SIZES_BYTES = {
        "vit_h": 2_300_000_000,
        "vit_l": 1_100_000_000,
        "vit_b": 300_000_000,
    }
    
    def __init__(self):
        self.model_dir = Path(settings.MODEL_CACHE_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, SamPredictor] = {}
        self.model_stats: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def get_model_info(self, model_type: str) -> Dict:
        """Get model information and status."""
        if model_type not in self.MODEL_CONFIG:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = self.MODEL_CONFIG[model_type]
        model_path = self.model_dir / config["filename"]
        
        info = {
            "model_type": model_type,
            "status": "not_downloaded",
            "file_size": config["size"],
            "description": config["description"],
            "file_path": str(model_path),
            "file_exists": model_path.exists(),
            "loaded": model_type in self.loaded_models,
            "memory_usage": None,
            "last_used": self.model_stats.get(model_type, {}).get("last_used"),
            "load_time": self.model_stats.get(model_type, {}).get("load_time"),
        }
        
        if model_path.exists():
            info["status"] = "downloaded"
            info["actual_size"] = model_path.stat().st_size
            
        if model_type in self.loaded_models:
            info["status"] = "loaded"
            info["memory_usage"] = self._get_model_memory_usage(model_type)
            
        return info
    
    async def download_model(self, model_type: str, progress_callback=None) -> bool:
        """Download model if not already present."""
        if model_type not in self.MODEL_CONFIG:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = self.MODEL_CONFIG[model_type]
        model_path = self.model_dir / config["filename"]
        
        if model_path.exists():
            min_size = self.MIN_SIZES_BYTES.get(model_type)
            try:
                actual = model_path.stat().st_size
                if min_size and actual >= min_size:
                    logger.info(f"Model {model_type} already downloaded")
                    return True
                else:
                    logger.warning(f"Existing {model_type} checkpoint is invalid (size={actual}); re-downloading")
                    model_path.unlink()
            except Exception:
                try:
                    model_path.unlink()
                except Exception:
                    pass
        
        async with self._lock:
            if model_path.exists():
                return True
            logger.info(f"Downloading {model_type} model from {config['url']}")
            attempts = 0
            last_err = None
            while attempts < 3:
                attempts += 1
                try:
                    timeout = aiohttp.ClientTimeout(total=3600)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(config["url"], headers={"User-Agent": "image-quarry"}) as response:
                            response.raise_for_status()
                            total_size = int(response.headers.get('content-length', 0))
                            downloaded = 0
                            async with aiofiles.open(model_path, 'wb') as file:
                                async for chunk in response.content.iter_chunked(1024 * 1024):
                                    if not chunk:
                                        continue
                                    await file.write(chunk)
                                    downloaded += len(chunk)
                                    if progress_callback and total_size > 0:
                                        progress = (downloaded / total_size) * 100
                                        await progress_callback(progress)
                            if total_size > 0 and downloaded < total_size:
                                raise aiohttp.ClientPayloadError(
                                    message=f"incomplete payload {downloaded}/{total_size}",
                                    history=None,
                                )
                    logger.info(f"Successfully downloaded {model_type} model")
                    return True
                except Exception as e:
                    last_err = e
                    logger.warning(f"Download attempt {attempts} failed for {model_type}: {str(e)}")
                    try:
                        if model_path.exists():
                            model_path.unlink()
                    except Exception:
                        pass
                    await asyncio.sleep(min(5 * attempts, 15))
            logger.error(f"Failed to download {model_type} model after retries: {str(last_err)}")
            if model_path.exists():
                try:
                    model_path.unlink()
                except Exception:
                    pass
            raise last_err
    
    async def load_model(self, model_type: str) -> SamPredictor:
        """Load model into memory."""
        if model_type not in self.MODEL_CONFIG:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Return cached model if available
        if model_type in self.loaded_models:
            self.model_stats[model_type]["last_used"] = datetime.utcnow()
            try:
                cpu_mb = self._get_process_memory_mb()
                gpu = self._get_model_memory_usage(model_type)
                logger.info(f"Reusing cached model model={model_type} location=ModelManager.load_model cpu_mb={cpu_mb} gpu={gpu}")
            except Exception:
                pass
            return self.loaded_models[model_type]
        
        async with self._lock:
            # Check again after acquiring lock
            if model_type in self.loaded_models:
                return self.loaded_models[model_type]
            
            # Ensure model is downloaded
            await self.download_model(model_type)
            
            config = self.MODEL_CONFIG[model_type]
            model_path = self.model_dir / config["filename"]
            
            logger.info(f"Loading {model_type} model into memory")
            start_time = datetime.utcnow()
            op_id = str(uuid.uuid4())
            cpu_mb_before = self._get_process_memory_mb()
            gpu_before = self._get_model_memory_usage(model_type)
            
            try:
                min_size = self.MIN_SIZES_BYTES.get(model_type)
                try:
                    actual = model_path.stat().st_size
                    if min_size and actual < min_size:
                        logger.warning(f"Checkpoint size too small for {model_type} (size={actual}); re-downloading")
                        if model_path.exists():
                            model_path.unlink()
                        await self.download_model(model_type)
                except Exception:
                    await self.download_model(model_type)
                # Load model
                model = sam_model_registry[model_type](checkpoint=str(model_path))
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    model = model.cuda()
                    logger.info(f"Loaded {model_type} model on GPU")
                else:
                    logger.info(f"Loaded {model_type} model on CPU")
                
                predictor = SamPredictor(model)
                
                # Cache model
                self.loaded_models[model_type] = predictor
                
                # Update stats
                load_time = (datetime.utcnow() - start_time).total_seconds()
                self.model_stats[model_type] = {
                    "load_time": load_time,
                    "last_used": datetime.utcnow(),
                    "loaded_at": datetime.utcnow(),
                }
                cpu_mb_after = self._get_process_memory_mb()
                gpu_after = self._get_model_memory_usage(model_type)
                logger.info(
                    f"Model load completed op_id={op_id} model={model_type} location=ModelManager.load_model duration={load_time:.2f}s cpu_before={cpu_mb_before}MB cpu_after={cpu_mb_after}MB gpu_before={gpu_before} gpu_after={gpu_after}"
                )
                
                logger.info(f"Successfully loaded {model_type} model in {load_time:.2f}s")
                return predictor
                
            except Exception as e:
                logger.error(f"Failed to load {model_type} model: {str(e)}")
                raise
    
    async def unload_model(self, model_type: str) -> bool:
        """Unload model from memory."""
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            if model_type in self.model_stats:
                del self.model_stats[model_type]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded {model_type} model from memory")
            return True
        
        return False
    
    async def get_loaded_models(self) -> Dict[str, Dict]:
        """Get information about all loaded models."""
        result = {}
        for model_type in self.loaded_models.keys():
            result[model_type] = await self.get_model_info(model_type)
        return result

    async def list_available_models(self) -> list:
        """List available SAM models with cache/load status."""
        models = []
        for model_type, cfg in self.MODEL_CONFIG.items():
            model_path = self.model_dir / cfg["filename"]
            cached = model_path.exists()
            loaded = model_type in self.loaded_models
            # Approximate sizes in MB from known values
            approx_size_mb = {
                "vit_h": 2400.0,
                "vit_l": 1200.0,
                "vit_b": 375.0,
            }.get(model_type, None)

            models.append({
                "model_type": model_type,
                "name": model_type,
                "description": cfg.get("description"),
                "size_mb": approx_size_mb,
                "cached": cached,
                "loaded": loaded,
                "download_url": cfg.get("url"),
                "last_used": self.model_stats.get(model_type, {}).get("last_used"),
            })
        return models
    
    def _get_model_memory_usage(self, model_type: str) -> Optional[str]:
        """Get memory usage for a loaded model."""
        if not torch.cuda.is_available():
            return None
        
        try:
            # Get GPU memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            
            return f"{memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
        except Exception:
            return None
    
    def _get_process_memory_mb(self) -> Optional[float]:
        try:
            import psutil
            p = psutil.Process(os.getpid())
            return float(p.memory_info().rss) / (1024**2)
        except Exception:
            return None
    
    async def cleanup(self):
        """Cleanup all loaded models."""
        logger.info("Cleaning up model manager")
        
        for model_type in list(self.loaded_models.keys()):
            await self.unload_model(model_type)
        
        self.loaded_models.clear()
        self.model_stats.clear()


# Global model manager instance
model_manager = ModelManager()
