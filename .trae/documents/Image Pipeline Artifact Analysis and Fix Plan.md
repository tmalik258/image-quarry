## Objective
- Preserve original transparency end‑to‑end. Never introduce a black background. Use enhanced image for segmentation while keeping alpha intact in outputs.

## Changes Focused on Alpha Preservation
1. Preprocessing for SAM (no background compositing):
   - Location: `app/services/image_processor.py:55-58`.
   - Action: if input is RGBA, produce `rgb_for_model = rgba[:, :, :3]` (3‑channel slice) for SAM; retain `image_rgba` as the authoritative image for saving/extraction. Do not composite onto any color.
   - Resizing rule: when resizing RGBA for enhancement or visualization, premultiply alpha only during resampling to avoid edge halos; otherwise, keep pixels unchanged.
2. Visualization save:
   - Location: `app/services/image_processor.py:278-285`.
   - Action: remove `convert('RGB')`; save the overlay as RGBA PNG to keep transparency.
3. Source image save:
   - Location: `app/services/background_tasks.py:546-549` and pipeline saves.
   - Action: save the original/enhanced RGBA image (not `image_rgb`). If image has no alpha, save as PNG/WebP lossless or JPEG per policy.
4. Mask decoding & edge cleanup:
   - Location: `app/services/image_processor.py:323-327`.
   - Action: resize masks with `INTER_NEAREST`, then apply `morphologyEx` open/close + optional `medianBlur(3)`; feather 1–2 px alpha border to prevent pixel loss at edges.
5. Object PNG writes:
   - Location: `app/services/object_extractor.py`.
   - Action: write objects via Pillow as RGBA to avoid BGR/RGB mix‑ups; preserve alpha.

## Handling Images Without Transparency
- If input lacks alpha, run SAM to derive background mask and keep background as‑is unless explicitly removed; do not force a black background. Outputs stay in the original mode.

## Validation
- Confirm transparency preserved (alpha channel present) in `source.png`, `overlay.png`, and object PNGs.
- Compare edge quality and stray pixel count before/after; run on enhanced images for segmentation.

## Deliverables
- Alpha‑preserving pipeline with enhanced image input for SAM, clean masks, and artifact‑free outputs.
- Technical report including file references, before/after metrics, and recommendations for further tuning.