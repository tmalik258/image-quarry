import os
import json
from PIL import Image
import numpy as np


def _make_sample_img(fmt: str, alpha: bool = False) -> bytes:
    w, h = 800, 1440
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # gradient + text-like lines
    for y in range(h):
        arr[y, :, 0] = (y * 255) // h
        arr[y, :, 1] = (255 - (y * 255) // h)
    img = Image.fromarray(arr, mode="RGB")
    if alpha:
        a = Image.new("L", (w, h), 255)
        img = Image.merge("RGBA", (*img.split(), a))
    from io import BytesIO
    b = BytesIO()
    img.save(b, format=fmt)
    return b.getvalue()


def test_quality_enhancement(tmp_path):
    from app.config import settings
    from app.services.quality import QualityEnhancer
    enh = QualityEnhancer(settings)
    job_dir = os.path.join(tmp_path, "job")
    os.makedirs(job_dir, exist_ok=True)
    data = _make_sample_img("PNG", alpha=True)
    out = enh.enhance(data, content_type="image/png", job_dir=job_dir)
    assert os.path.exists(out["path"])  # enhanced source
    assert os.path.exists(out["report_path"])  # quality.json
    assert os.path.exists(out["diff_path"])  # diff.png
    with open(out["report_path"], "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["enhanced_size_bytes"] >= settings.QUALITY_TARGET_BYTES
    assert rep["dimensions_after"][0] >= settings.QUALITY_MIN_WIDTH
    assert rep["dimensions_after"][1] >= settings.QUALITY_MIN_HEIGHT