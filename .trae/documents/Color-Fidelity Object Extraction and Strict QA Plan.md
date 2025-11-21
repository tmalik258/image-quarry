## Goals
- Preserve exact color from source/enhanced images in objects/* outputs.
- Keep ICC/color profiles in object saves.
- Tighten extraction parameters and add automated color QA.

## Changes Overview
1. Use RGBA for extraction; never use premultiplied RGB for object pixels.
2. Remove channel swaps; keep original channel order when building object RGBA.
3. Preserve ICC profile when saving object PNG/WebP.
4. Add color QA: histogram and ΔE checks; log and flag deviations.
5. Expose stricter segmentation/extraction parameters and document where to set them.

## Code Edits (exact locations)
- `app/services/object_extractor.py`
  1) Update object composition to keep original colors:
     - Function: `ObjectExtractor.extract`
     - Lines: 31–36 currently split/merge channels
     - Change: build `rgba = np.dstack([cropped_image[:, :, 0], cropped_image[:, :, 1], cropped_image[:, :, 2], cropped_mask])` (no swapping) and operate on `image_rgba` input.
  2) Save with ICC and Pillow:
     - Function: `save_objects`
     - Lines: 39–47
     - Add arg `icc_profile: Optional[bytes] = None`; call `Image.fromarray(rgba).save(out_path, format="PNG", icc_profile=icc_profile)`.

- `app/services/background_tasks.py`
  3) Use RGBA for extraction and pass ICC:
     - Function: `_segment_pipeline_async`
     - Lines: 528–545
     - Change: obtain `orig_pil = Image.open(io.BytesIO(image_bytes))`; `icc = orig_pil.info.get("icc_profile")`; call `objects = extractor.extract(image_rgba, decoded_masks)` then `extractor.save_objects(objects, object_dir, icc_profile=icc)`.
     - Function: `_extract_objects_async`
     - Lines: 419–429
     - Change similarly: use `image_rgba` and pass ICC to `save_objects`.

- `app/services/pipeline.py`
  4) Use RGBA for extraction and ICC in saves:
     - Lines: 126–131
     - Change: call `extractor.extract(image_rgba, decoded)` and `extractor.save_objects(..., icc_profile=orig_pil.info.get("icc_profile"))` (load `orig_pil` near start with `Image.open(io.BytesIO(image_bytes))`).

- `app/services/quality.py` (QA utilities)
  5) Add color QA helpers:
     - New functions:
       - `histogram_rgb(arr: np.ndarray) -> Dict[str, np.ndarray]` (per-channel histograms)
       - `delta_e(source: np.ndarray, target: np.ndarray) -> float` (convert to LAB, compute mean ΔE)
       - `compare_color(source_rgba, object_rgba, mask) -> Dict[str, float]` returning `deltaE`, per-channel histogram diffs
     - Log deviations > thresholds; thresholds from settings.

- `app/services/image_processor.py`
  6) Keep existing `preprocess_image_pair` for RGBA+model RGB. No further change needed for color except use of RGBA in extraction as above.

## Parameters to Tune (where)
- Segmentation (global defaults): `app/services/pipeline.py:79–86` `auto_params`
  - Set: `stability_score_thresh=0.95`, `pred_iou_thresh=0.92`, `min_mask_region_area=4000`, `points_per_side=32 (GPU)`.
- Segmentation (per-job overrides):
  - Background pipeline: `app/services/background_tasks.py:486–494` (keys in `params`)
  - Object extraction task: `app/services/background_tasks.py:409–415` (keys in `params`)
- New extraction QA thresholds in settings:
  - File: `app/config.py`
  - Add:
    - `EXTRACTION_MAX_DELTA_E: float = 2.0`
    - `EXTRACTION_MAX_HISTO_DIFF: float = 0.02`
    - `EXTRACTION_EDGE_FEATHER_PX: int = 1`

## Quality Control & Logging
- During extraction (pipeline/background):
  - For each object, compute `deltaE` and histogram differences using `quality.compare_color(...)` between source region and object.
  - Log deviations with job_id, object index, and metrics; flag if `deltaE > EXTRACTION_MAX_DELTA_E` or histogram diff exceeds threshold.
  - Save a small JSON per object under `objects/obj_{i}_color_report.json` with metrics.

## Color Spaces Handling
- Loading: always convert to `RGBA` via Pillow (`Image.convert("RGBA")`) to correctly handle CMYK/embedded ICC.
- Saving: include `icc_profile` on PNG/WebP where present; JPEG for non-alpha with `exif` and `icc_profile`.

## Tests / Verification
- Add unit tests in `app/tests/` that:
  - Load a transparent PNG with ICC, run extraction, assert `deltaE <= EXTRACTION_MAX_DELTA_E` and per-channel histo diff ≤ threshold.
  - Verify objects retain alpha and color profile metadata (`icc_profile` present).
- Add logs: deviations reported in application logs and aggregated at job end.

## Documentation (developer guide)
- Where to tune segmentation: pipeline defaults (`app/services/pipeline.py:79–86`), per-job params in background tasks, and generator defaults (`app/services/image_processor.py:69–75`).
- Where to set QA thresholds: `app/config.py` fields listed above.
- What changed for color: extraction now uses RGBA source pixels and preserves ICC; saving uses Pillow with `icc_profile`.

## Rollout Plan
- Implement changes and run on your shared sample set.
- Inspect `objects/*` color visually and via reports; adjust thresholds/params as needed.
- Monitor logs for flags; iterate model params only if artifacts persist after color fixes.
