import base64
import io
from typing import List, Dict, Any, Optional, Tuple
import os
from PIL import Image, ImageDraw
from app.utils.logging import setup_logger
import numpy as np
import cv2
from segment_anything import SamPredictor, SamAutomaticMaskGenerator
import supervision as sv
import asyncio

from app.config import settings

logger = setup_logger(__name__)


class ImageProcessor:
    """Handles image processing and SAM segmentation."""
    
    def __init__(self):
        self.supported_formats = {
            'image/jpeg': 'JPEG',
            'image/png': 'PNG',
            'image/webp': 'WEBP'
        }
    
    def validate_image(self, image_data: bytes, content_type: str) -> Tuple[bool, str]:
        """Validate image format and size."""
        if content_type not in self.supported_formats:
            return False, f"Unsupported format: {content_type}. Supported: {list(self.supported_formats.keys())}"
        
        max_bytes = int(settings.MAX_IMAGE_SIZE_MB) * 1024 * 1024
        if len(image_data) > max_bytes:
            return False, f"Image too large: {len(image_data)} bytes. Maximum: {max_bytes} bytes"
        
        try:
            # Try to open image
            image = Image.open(io.BytesIO(image_data))
            
            # Check dimensions
            if image.width > settings.MAX_IMAGE_DIMENSION or image.height > settings.MAX_IMAGE_DIMENSION:
                return False, f"Image dimensions too large: {image.width}x{image.height}. Maximum: {settings.MAX_IMAGE_DIMENSION}x{settings.MAX_IMAGE_DIMENSION}"
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        arr = np.array(image)
        logger.info(f"Preprocessed image: {arr.shape}")
        return arr

    def preprocess_image_pair(self, image_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        arr_rgba = np.array(image)
        arr_rgb = self.rgba_to_rgb_for_model(arr_rgba)
        logger.info(f"Preprocessed image RGBA: {arr_rgba.shape} RGB: {arr_rgb.shape}")
        return arr_rgba, arr_rgb

    def rgba_to_rgb_for_model(self, rgba: np.ndarray, bg_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
        a = rgba[:, :, 3].astype(np.float32) / 255.0
        rgb = rgba[:, :, :3].astype(np.float32)
        bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        out = rgb * a[..., None] + bg * (1.0 - a[..., None])
        return out.clip(0, 255).astype(np.uint8)
    
    async def generate_masks(
        self,
        predictor: SamPredictor,
        image: np.ndarray,
        points_per_side: int = 64,
        pred_iou_thresh: float = 0.90,
        stability_score_thresh: float = 0.92,
        crop_n_layers: int = 1,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 1000,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            import time
            import uuid
            predictor.set_image(image)
            logger.info(
                f"Generator params job_id={job_id} points_per_side={points_per_side} pred_iou_thresh={pred_iou_thresh} "
                f"stability_score_thresh={stability_score_thresh} crop_n_layers={crop_n_layers} "
                f"crop_n_points_downscale_factor={crop_n_points_downscale_factor} min_mask_region_area={min_mask_region_area}"
            )
            generator = SamAutomaticMaskGenerator(
                model=predictor.model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers=crop_n_layers,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_mask_region_area=min_mask_region_area,
            )
            op_id = str(uuid.uuid4())
            start = time.perf_counter()
            done = False
            async def heartbeat():
                while not done:
                    try:
                        import psutil
                        rss_mb = float(psutil.Process(os.getpid()).memory_info().rss) / (1024**2)
                        import torch
                        if torch.cuda.is_available():
                            mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                            mem_res = torch.cuda.memory_reserved() / (1024**2)
                            logger.info(
                                f"HB op={op_id} job={job_id if job_id else 'none'} dur={time.perf_counter()-start:.2f}s rss={rss_mb:.0f}MB gpu={mem_alloc:.0f}/{mem_res:.0f}MB"
                            )
                        else:
                            logger.info(
                                f"HB op={op_id} job={job_id if job_id else 'none'} dur={time.perf_counter()-start:.2f}s rss={rss_mb:.0f}MB gpu=NA"
                            )
                    except Exception:
                        pass
                    await asyncio.sleep(2)
            hb_task = asyncio.create_task(heartbeat())
            try:
                logger.info(f"Generator start job_id={job_id} op_id={op_id}")
                sam_result = await asyncio.to_thread(generator.generate, image)
                logger.info(f"Generator done job_id={job_id} op_id={op_id} elapsed={time.perf_counter()-start:.2f}s masks={len(sam_result)}")
            finally:
                done = True
                try:
                    hb_task.cancel()
                except Exception:
                    pass
            masks: List[Dict[str, Any]] = []
            for i, m in enumerate(sam_result):
                masks.append({
                    "segmentation": self._mask_to_base64(m["segmentation"]),
                    "area": int(m["area"]),
                    "bbox": m["bbox"],
                    "predicted_iou": float(m["predicted_iou"]),
                    "stability_score": float(m["stability_score"]),
                    "point_coords": m.get("point_coords"),
                    "crop_box": m.get("crop_box"),
                })
            return masks
        except Exception as e:
            logger.error(f"Failed to generate masks: {str(e)}")
            raise

    async def segment_image(
        self,
        image_data: bytes,
        model_type: str,
        points: Optional[List[Dict[str, float]]] = None,
        boxes: Optional[List[Dict[str, float]]] = None,
        multimask_output: bool = True,
        auto_params: Optional[Dict[str, Any]] = None,
        predictor: Optional[SamPredictor] = None,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from time import perf_counter
        t0 = perf_counter()
        image_rgba, image_rgb = self.preprocess_image_pair(image_data)
        if predictor is None:
            from app.services.model_manager import model_manager
            predictor = await model_manager.load_model(model_type or settings.DEFAULT_MODEL)
        logger.info(f"Segmentation started job_id={job_id}")
        if boxes or points:
            predictor.set_image(image_rgb)
            mask_items: List[Dict[str, Any]] = []
            logger.info(f"Progress 25% job_id={job_id} step=points_boxes_setup")
            if boxes:
                for b in boxes:
                    h, w = image_rgb.shape[0], image_rgb.shape[1]
                    x = int(b.get("x", b.get("x1", 0)) * w) if 0 <= b.get("x", b.get("x1", 0)) <= 1 else int(b.get("x", b.get("x1", 0)))
                    y = int(b.get("y", b.get("y1", 0)) * h) if 0 <= b.get("y", b.get("y1", 0)) <= 1 else int(b.get("y", b.get("y1", 0)))
                    width = int(b.get("width", max(0, b.get("x2", 0) - b.get("x1", 0))) * w) if b.get("width") is not None else int(max(0, b.get("x2", 0) - b.get("x1", 0)) * w) if 0 <= b.get("x1", 0) <= 1 else int(max(0, b.get("x2", 0) - b.get("x1", 0)))
                    height = int(b.get("height", max(0, b.get("y2", 0) - b.get("y1", 0))) * h) if b.get("height") is not None else int(max(0, b.get("y2", 0) - b.get("y1", 0)) * h) if 0 <= b.get("y1", 0) <= 1 else int(max(0, b.get("y2", 0) - b.get("y1", 0)))
                    box_xyxy = np.array([x, y, x + width, y + height])
                    masks, scores, logits = predictor.predict(box=box_xyxy, multimask_output=multimask_output)
                    for idx, m in enumerate(masks):
                        mask_items.append({
                            "segmentation": self._mask_to_base64(m),
                            "area": int(np.sum(m.astype(np.uint8))),
                            "bbox": [int(x), int(y), int(width), int(height)],
                            "predicted_iou": float(scores[idx]) if scores is not None else 1.0,
                            "stability_score": float(scores[idx]) if scores is not None else 1.0,
                        })
            logger.info(f"Segmentation with boxes produced {len(mask_items)} masks job_id={job_id}")
            return mask_items
        params = auto_params or {}
        logger.info(f"Progress 25% job_id={job_id} step=set_image")
        import torch
        h, w = image_rgb.shape[0], image_rgb.shape[1]
        effective_params = dict(params)
        if torch.cuda.is_available():
            requested_pps = int(effective_params.get("points_per_side", 64))
            effective_params["points_per_side"] = min(requested_pps, 32)
        else:
            effective_params["points_per_side"] = int(effective_params.get("points_per_side", 16))
        logger.info(
            f"Progress 50% job_id={job_id} step=generator_init image={w}x{h} points_per_side={effective_params['points_per_side']} "
            f"pred_iou_thresh={float(params.get('pred_iou_thresh', 0.90))} stability_score_thresh={float(params.get('stability_score_thresh', 0.92))} "
            f"crop_n_layers={int(params.get('crop_n_layers', 1))} crop_n_points_downscale_factor={int(params.get('crop_n_points_downscale_factor', 2))} "
            f"min_mask_region_area={int(params.get('min_mask_region_area', 1000))}"
        )
        masks = await self.generate_masks(
            predictor,
            image_rgb,
            points_per_side=int(effective_params.get("points_per_side", 64)),
            pred_iou_thresh=float(params.get("pred_iou_thresh", 0.90)),
            stability_score_thresh=float(params.get("stability_score_thresh", 0.92)),
            crop_n_layers=int(params.get("crop_n_layers", 1)),
            crop_n_points_downscale_factor=int(params.get("crop_n_points_downscale_factor", 2)),
            min_mask_region_area=int(params.get("min_mask_region_area", 1000)),
            job_id=job_id,
        )
        logger.info(f"Progress 75% job_id={job_id} step=generator_generate")
        logger.info(f"Progress 90% job_id={job_id} step=masks_generated count={len(masks)}")
        try:
            import psutil
            rss_mb = float(psutil.Process(os.getpid()).memory_info().rss) / (1024**2)
            logger.info(f"Resource snapshot location=ImageProcessor.segment_image stage=post_generate rss_mb={rss_mb:.2f} job_id={job_id}")
        except Exception:
            pass
        logger.info(f"Segmentation completed in {perf_counter() - t0:.2f}s with {len(masks)} masks job_id={job_id}")
        return masks
    
    def create_visualization(
        self,
        original_image: np.ndarray,
        masks: List[Dict[str, Any]],
        max_masks: int = 10,
        with_labels: bool = True
    ) -> str:
        """Create visualization overlay of masks on original image."""
        try:
            # Convert original image to PIL
            original_pil = Image.fromarray(original_image)
            
            # Create overlay
            overlay = Image.new('RGBA', original_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Sort masks by area (largest first)
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            
            # Take top masks
            top_masks = sorted_masks[:max_masks]
            
            # Generate colors
            colors = self._generate_colors(len(top_masks))
            
            for i, mask_data in enumerate(top_masks):
                mask = self._base64_to_mask(mask_data['segmentation'], original_image.shape[:2])
                try:
                    mask = self.refine_mask(mask)
                except Exception:
                    pass
                
                # Convert mask to PIL
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                
                # Create colored mask
                color_overlay = Image.new('RGBA', original_pil.size, colors[i] + (128,))  # 50% transparency
                
                # Apply mask to overlay
                overlay = Image.composite(color_overlay, overlay, mask_pil)
                if with_labels:
                    try:
                        ys, xs = np.where(mask > 0)
                        if len(xs) and len(ys):
                            cx = int(np.mean(xs))
                            cy = int(np.mean(ys))
                            label_text = str(i + 1)
                            tw, th = 16, 16
                            bg = Image.new('RGBA', (tw, th), (0, 0, 0, 128))
                            overlay.paste(bg, (max(0, cx - tw // 2), max(0, cy - th // 2)))
                            draw.text((cx - 4, cy - 8), label_text, fill=(255, 255, 255, 255))
                    except Exception:
                        pass
            
            original_rgba = original_pil.convert('RGBA')
            result = Image.alpha_composite(original_rgba, overlay)
            buffer = io.BytesIO()
            result.save(buffer, format='PNG')
            buffer.seek(0)
            
            visualization_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            logger.info("Created visualization")
            return visualization_b64
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            raise
    
    def _mask_to_base64(self, mask: np.ndarray) -> str:
        """Convert mask to base64 encoded PNG."""
        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # Convert to PIL
        mask_pil = Image.fromarray(binary_mask, mode='L')
        
        # Save to buffer
        buffer = io.BytesIO()
        mask_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convert to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _base64_to_mask(self, base64_str: str, shape: Tuple[int, int]) -> np.ndarray:
        image_data = base64.b64decode(base64_str)
        mask_pil = Image.open(io.BytesIO(image_data))
        mask_array = np.array(mask_pil)
        if mask_array.shape != shape:
            mask_array = cv2.resize(mask_array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask_array > 127).astype(np.uint8)

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        k = np.ones((3, 3), np.uint8)
        m = (mask > 0).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        m = cv2.medianBlur(m, 3)
        return (m > 127).astype(np.uint8)
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        colors = []
        for i in range(num_colors):
            # Use HSV to generate distinct colors
            hue = (i * 360 / num_colors) % 360
            # Convert HSV to RGB (simplified)
            if hue < 60:
                r, g, b = 255, int(hue * 255 / 60), 0
            elif hue < 120:
                r, g, b = int((120 - hue) * 255 / 60), 255, 0
            elif hue < 180:
                r, g, b = 0, 255, int((hue - 120) * 255 / 60)
            elif hue < 240:
                r, g, b = 0, int((240 - hue) * 255 / 60), 255
            elif hue < 300:
                r, g, b = int((hue - 240) * 255 / 60), 0, 255
            else:
                r, g, b = 255, 0, int((360 - hue) * 255 / 60)
            
            colors.append((r, g, b))
        return colors
