import io
import os
import json
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import cv2


def _to_uint16(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        return img
    return (img.astype(np.uint16) * 257).clip(0, 65535)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return (img / 257.0).clip(0, 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a8 = a.astype(np.float32)
    b8 = b.astype(np.float32)
    mse = np.mean((a8 - b8) ** 2)
    if mse == 0:
        return 100.0
    PIX_MAX = 255.0
    return 20 * np.log10(PIX_MAX) - 10 * np.log10(mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Simple SSIM for grayscale; extend to RGB by averaging channels
    def _ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        K1, K2 = 0.01, 0.03
        L = 255.0
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
        s = num / (den + 1e-8)
        return float(np.mean(s))
    if a.ndim == 2:
        return _ssim_gray(a, b)
    # RGB: average SSIM per channel
    vals = []
    for c in range(a.shape[2]):
        vals.append(_ssim_gray(a[:, :, c], b[:, :, c]))
    return float(np.mean(vals))


class QualityEnhancer:
    def __init__(self, settings):
        self.settings = settings

    def _load_pil(self, data: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(data))
        return img

    def _pil_to_np(self, img: Image.Image) -> np.ndarray:
        if img.mode in ("RGB", "RGBA"):
            return np.array(img)
        return np.array(img.convert("RGBA"))

    def _np_to_pil(self, arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
        return Image.fromarray(arr)

    def _resize16(self, arr16: np.ndarray, scale: float) -> np.ndarray:
        h, w = arr16.shape[:2]
        nh, nw = int(round(h * scale)), int(round(w * scale))
        if arr16.ndim == 3:
            if arr16.shape[2] == 4:
                rgb16 = arr16[:, :, :3]
                a16 = arr16[:, :, 3]
                rgb16_res = cv2.resize(rgb16, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
                a16_res = cv2.resize(a16, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
                return np.dstack([rgb16_res, a16_res])
            return cv2.resize(arr16, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        return cv2.resize(arr16, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    def _save_with_metadata(self, pil_img: Image.Image, fmt: str, orig: Image.Image, out_path: str) -> Tuple[str, int]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        exif = None
        try:
            exif = orig.getexif()
        except Exception:
            exif = None
        icc = orig.info.get("icc_profile", None)
        params: Dict[str, Any] = {}
        if icc:
            params["icc_profile"] = icc
        if fmt.upper() == "PNG":
            pil_img.save(out_path, format="PNG", **params)
        elif fmt.upper() == "WEBP":
            params.update({"lossless": True, "quality": 100})
            pil_img.save(out_path, format="WEBP", **params)
        elif fmt.upper() in ("JPEG", "JPG"):
            params.update({"quality": max(90, 95), "subsampling": 0, "optimize": True})
            if exif:
                params["exif"] = exif.tobytes()
            pil_img.save(out_path, format="JPEG", **params)
        else:
            pil_img.save(out_path, format=fmt.upper(), **params)
        return out_path, os.path.getsize(out_path)

    def enhance(self, image_bytes: bytes, content_type: str, job_dir: str) -> Dict[str, Any]:
        src = self._load_pil(image_bytes)
        orig_np = self._pil_to_np(src)
        h0, w0 = orig_np.shape[:2]
        min_w = getattr(self.settings, "QUALITY_MIN_WIDTH", 800)
        min_h = getattr(self.settings, "QUALITY_MIN_HEIGHT", 1440)
        scale0 = max(min_w / w0, min_h / h0, getattr(self.settings, "QUALITY_SCALE_FACTOR", 1.5))
        arr16 = _to_uint16(orig_np)
        up16 = self._resize16(arr16, scale0)
        up8 = _to_uint8(up16)
        pil_up = self._np_to_pil(up8)
        # Choose format based on alpha
        has_alpha = (up8.ndim == 3 and up8.shape[2] == 4)
        fmt_priority = getattr(self.settings, "QUALITY_SAVE_FORMAT_PRIORITY", ["PNG", "WEBP", "JPEG"])
        fmt = fmt_priority[0] if has_alpha else (fmt_priority[-1])
        out_path = os.path.join(job_dir, "source_enhanced." + ("png" if fmt == "PNG" else ("webp" if fmt == "WEBP" else "jpg")))
        path, size_bytes = self._save_with_metadata(pil_up, fmt, src, out_path)
        target_bytes = getattr(self.settings, "QUALITY_TARGET_BYTES", 1_000_000)
        # If not large enough, increase scale iteratively
        s = scale0
        tries = 0
        while size_bytes < target_bytes and tries < 3:
            s *= 1.25
            up16 = self._resize16(arr16, s)
            up8 = _to_uint8(up16)
            pil_up = self._np_to_pil(up8)
            path, size_bytes = self._save_with_metadata(pil_up, fmt, src, out_path)
            tries += 1
        # QA metrics: compare enhanced downscaled to original size
        ds_enh = cv2.resize(up8[:, :, :3] if up8.ndim == 3 else up8, (w0, h0), interpolation=cv2.INTER_AREA)
        base_rgb = orig_np[:, :, :3] if orig_np.ndim == 3 else orig_np
        _psnr = psnr(ds_enh, base_rgb)
        _ssim = ssim(ds_enh, base_rgb)
        # Visual diff heatmap
        diff = cv2.absdiff(ds_enh, base_rgb)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if diff.ndim == 3 else diff
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        heat_path = os.path.join(job_dir, "diff.png")
        Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).save(heat_path)
        # Artifact heuristics
        edges_after = cv2.Canny(ds_enh.astype(np.uint8), 100, 200)
        lines = cv2.HoughLinesP(edges_after, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        artifacts = {"lines": bool(lines is not None and len(lines) > 0), "edge_pixel_loss": False}
        # Persist report
        report = {
            "original_size_bytes": len(image_bytes),
            "enhanced_size_bytes": size_bytes,
            "dimensions_before": [w0, h0],
            "dimensions_after": [pil_up.width, pil_up.height],
            "psnr": _psnr,
            "ssim": _ssim,
            "artifacts_detected": artifacts,
            "format": fmt,
        }
        with open(os.path.join(job_dir, "quality.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # Return enhanced bytes for downstream steps
        buf = io.BytesIO()
        save_fmt = "PNG" if fmt == "PNG" else ("WEBP" if fmt == "WEBP" else "JPEG")
        pil_up.save(buf, format=save_fmt)
        buf.seek(0)
        return {
            "path": path,
            "bytes": buf.getvalue(),
            "report_path": os.path.join(job_dir, "quality.json"),
            "diff_path": heat_path,
        }