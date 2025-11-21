import os
import io
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from app.utils.logging import setup_logger

logger = setup_logger(__name__)

class ObjectExtractor:
    def __init__(self, min_pixels: int = 5000):
        self.min_pixels = int(min_pixels)

    def extract(self, image_rgba: np.ndarray, sam_result: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        masks = [mask["segmentation"] if isinstance(mask["segmentation"], np.ndarray) else mask["segmentation"] for mask in sam_result]
        objects: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        for m in sam_result:
            mask = m["segmentation"]
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            mask_bin = (mask > 0).astype(np.uint8) * 255
            if int(np.sum(mask_bin)) < self.min_pixels:
                continue
            y, x = np.where(mask_bin > 0)
            if len(x) == 0 or len(y) == 0:
                continue
            x_min, x_max = int(x.min()), int(x.max())
            y_min, y_max = int(y.min()), int(y.max())
            cropped_mask = mask_bin[y_min:y_max, x_min:x_max]
            cropped_image = image_rgba[y_min:y_max, x_min:x_max]
            segmented_object = cv2.bitwise_and(cropped_image[:, :, :3], cropped_image[:, :, :3], mask=cropped_mask)
            try:
                alpha = cv2.GaussianBlur(cropped_mask, (3, 3), 0)
            except Exception:
                alpha = cropped_mask
            rgba = np.dstack([segmented_object[:, :, 0], segmented_object[:, :, 1], segmented_object[:, :, 2], alpha])
            objects.append((rgba, (x_min, y_min, x_max, y_max)))
        return objects

    def save_objects(self, objects: List[np.ndarray], save_dir: str, icc_profile: Optional[bytes] = None) -> List[str]:
        os.makedirs(save_dir, exist_ok=True)
        saved_paths: List[str] = []
        for i, obj in enumerate(objects, start=1):
            out_path = os.path.join(save_dir, f"object_{i}.png")
            try:
                Image.fromarray(obj).save(out_path, format="PNG", icc_profile=icc_profile)
            except Exception:
                Image.fromarray(obj).save(out_path, format="PNG")
            saved_paths.append(out_path)
        logger.info(f"Saved segmented objects count={len(saved_paths)} dir={save_dir}")
        return saved_paths

