## Overview

Enhance the pipeline to upscale and preserve image quality end-to-end, add objective QA metrics (PSNR/SSIM), visual diffs, metadata/color profile retention, and rigorous tests. Integrate as an optional pre-segmentation step with configurable thresholds.

## Components

* Service: `app/services/quality.py`

  * High-bit-depth processing (16-bit/channel) utilities

  * Upscaling (Lanczos/Bicubic) and optional AI SR (pluggable)

  * Color profile (ICC) and EXIF/metadata preservation

  * Alpha channel-safe conversions (RGBA/LA/WebP)

  * File-size targeting and save strategies (lossless/quality ≥90)

  * QA metrics: PSNR, SSIM, pixel diff, visual diff heatmaps

* Pipeline Integration: `app/services/pipeline.py`

  * Pre-segmentation enhancement step: reads uploaded bytes, processes at 16-bit, saves `source_enhanced.(png|webp|jpg)`

  * Writes QA report `quality.json` and visual diff `diff.png`

  * Passes 8-bit normalized RGB (no profile loss) to SAM for segmentation

* Background Tasks: `app/services/background_tasks.py`

  * Run enhancement and QA inside `segment_pipeline_task` prior to segmentation

* Router: `app/routes/jobs.py`

  * Include `source_enhanced`, `quality.json`, and `diff.png` in `/jobs/{job_id}/files`

## Upscaling Strategy

* Default: classical upscaling

  * Resize using PIL (Lanczos) or OpenCV (INTER\_LANCZOS4), preserve aspect ratio

  * Ensure minimum dimensions ≥ `800x1440`; scale proportionally (e.g., 1.5–2×) to exceed target file size

* Optional: AI super-resolution (feature-flag)

  * Plug-in support for OpenCV dnn\_superres (EDSR/MDSR) or Real-ESRGAN (Torch)

  * Caches the SR model; falls back to Lanczos when SR unavailable

## Source Preservation & Saving

* Maintain identical visual quality (no new artifacts)

  * Intermediate processing in 16-bit arrays (`np.uint16`)

  * Avoid unnecessary color-space conversions; retain alpha channel

* Save rules (format-aware)

  * PNG: lossless, compress level moderate, preserve alpha and ICC

  * WebP: `lossless=True`, `quality=100` for alpha images; otherwise `quality>=90`

  * JPEG: `quality>=90`, `subsampling=0`, preserve ICC, EXIF, `optimize=True`

* Metadata retention

  * Read `image.info['icc_profile']` and `getexif()`; write back on save

## Quality Control & Metrics

* Pre/post validation

  * Pixel-by-pixel comparison on common space (convert both to 8-bit RGB for metrics)

  * PSNR: implement closed-form function

  * SSIM: implement structural similarity (windowed mean/luminance/contrast) without external deps

* Visual diff

  * Absolute difference heatmap overlaid on source; save as `diff.png`

* Report

  * `quality.json`: `{original_size_bytes, enhanced_size_bytes, dimensions_before, dimensions_after, psnr, ssim, artifacts_detected: {lines: bool, edge_pixel_loss: bool}}`

## Artifacts Detection

* Unwanted lines

  * Simple edge-map scan for long, thin high-contrast lines post-enhancement; compare deltas vs original

* Edge pixel loss

  * Per-object contour dilation check on mask boundaries; compute boundary preservation score

## Technical Details

* 16-bit processing

  * Convert PIL image -> `np.uint16` via scaling; perform resize in OpenCV or PIL that supports higher bit-depth

* Color profiles

  * Preserve ICC profile bytes through PIL `save(icc_profile=icc)`

* Alpha channel

  * Maintain RGBA for PNG/WebP; avoid `convert('RGB')` unless strictly needed for SAM (segmentation feed uses RGB, but preserves a separate RGBA copy for saving)

* File-size targeting

  * After enhancement, select format/quality and adjust scale/quality until `>= min_file_size_bytes` (configurable; target >1MB)

## Configuration

* `settings` additions

  * `QUALITY_MIN_WIDTH=800`, `QUALITY_MIN_HEIGHT=1440`

  * `QUALITY_SCALE_FACTOR=1.5` (default)

  * `QUALITY_TARGET_BYTES=1_000_000`

  * `QUALITY_ENABLE_SR=False`

  * `QUALITY_SAVE_FORMAT_PRIORITY=['PNG','WEBP','JPEG']`

## Logging

* QA logs

  * `QualityEnhancer: dims_before=WxH dims_after=WxH psnr=X ssim=Y target_bytes=Z actual_bytes=Z2 format=FMT`

* Heartbeat remains simplified as implemented

## Testing

* Add `app/tests/test_quality.py`

  * Cases: JPEG/PNG/WebP; fine details, gradients, text; transparent PNG/WebP

  * Assertions: size increase (>1MB), dims ≥ minimum, PSNR/SSIM above thresholds, metadata exists, alpha preserved, timing metrics

* Performance measurement

  * Record enhancement time, SR model load time, and overall pipeline impact

## Changes Summary

* New file: `app/services/quality.py`

* Update: integrate into `segment_pipeline_task` and `process_image_pipeline`

* Update: `jobs` route to expose new files (`source_enhanced`, `quality.json`, `diff.png`)

* Config updates in `settings`

* Tests under `app/tests/`

## Confirmation

On approval, I will implement the `quality.py` service, integrate the enhancement + QA into the pipeline and background tasks, update routes and settings, and deliver tests. I’ll keep AI SR pluggable (default off) and rely on Lanczos/Bicubic by default to meet the requirement while safeguarding performance.
