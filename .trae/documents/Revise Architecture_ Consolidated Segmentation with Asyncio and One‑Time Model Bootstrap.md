## Summary

* Remove the legacy model routes and all existing segmentation routes; introduce a single consolidated `/segment` endpoint.

* Implement a one‑time model bootstrap aligned with the notebook (GitHub install of `segment-anything`) plus first‑run weights download and caching.

* Choose Asyncio for processing: offload CPU/GPU steps to threads, manage the event loop, add robust error recovery.

* Deliver a complete pipeline: segmentation → masks → object extraction → multi‑object handling → organized file outputs.

## Deletions & Router Cleanup

* Delete `app/routes/models.py` entirely.

* Replace `app/routes/segment.py` with a new `app/routes/segment.py` exposing only `POST /segment`.

* Update `app/main.py` to remove the model router inclusion and update API info:

  * Remove `app.include_router(models.router)` (app/main.py:55).

  * Update `/info` response to remove `/models/` and legacy `/segment/*` entries (app/main.py:176–191).

## One‑Time Model Bootstrap

* Dependency install (from notebook): pin `segment-anything` via VCS URL in `requirements.txt`: `git+https://github.com/facebookresearch/segment-anything.git@<commit>`.

* Add `app/lib/model_bootstrap.py` with `async def ensure_model_ready(model_type: str)`:

  * Validate `import segment_anything` succeeds; if not, raise an explicit startup error with remediation steps.

  * Ensure weights exist in `settings.MODEL_CACHE_DIR`; on first run, download once via `aiohttp` to the correct filename (reuse existing `ModelManager` download logic).

  * Verify readable access and non‑zero file size; log outcome.

* Call `await ensure_model_ready(settings.DEFAULT_MODEL)` in the app lifespan startup (app/main.py:24–41).

## Consolidated Endpoint (Asyncio)

* New `@router.post("/segment")` in `app/routes/segment.py`:

  * Accept `image: UploadFile` via form‑data.

  * Validate MIME type and size using an improved `ImageProcessor.validate_image` (fix to use `settings.MAX_IMAGE_SIZE_MB` converted to bytes).

  * Process within a single request using Asyncio:

    * Use `await asyncio.to_thread(...)` to run CPU/GPU‑bound SAM predictor steps and OpenCV/PIL operations.

    * Ensure the event loop remains responsive; avoid blocking.

  * Return a structured JSON with saved file paths and metadata.

## Processing Pipeline

* Steps implemented in `app/services/pipeline.py` (≤200 LOC):

  1. Read image bytes and validate.
  2. Preprocess to RGB `np.ndarray`.
  3. Lazy‑load predictor once (reuse `ModelManager.load_model` cache); if loading fails, attempt a single retry after clearing CUDA cache.
  4. Generate masks automatically (mirror notebook’s `SamAutomaticMaskGenerator` defaults).
  5. Extract RGBA objects for each mask (`ObjectExtractor`), skip small masks (`min_pixels`).
  6. Save outputs under `settings.RESULTS_DIR/<timestamp>_<uuid>/`:

     * `overlay.png` (visualization of selected masks),

     * `mask_{i}.png` (base64 decoded → PNG),

     * `object_{i}.png` (RGBA crops),

     * `metadata.json` (sizes, counts, parameters, timings).
  7. Validate each file exists and is non‑zero before reporting.
  8. Cleanup tmp buffers; keep only the organized outputs.

## Error Handling & Recovery

* Input validation errors → 400 with JSON error code `INVALID_IMAGE`.

* Predict/load errors → 502/500 with codes `MODEL_LOAD_FAILED` or `SEGMENTATION_FAILED`.

* File I/O validation: check existence and size; if invalid, remove and record under `skipped`.

* Recovery:

  * If CUDA OOM, unload model, clear `torch.cuda.empty_cache()`, retry once.

  * If predictor is corrupted, rebuild predictor from cached weights.

* All errors logged via `setup_logger` with structured fields.

## Logging & Observability

* Structured logs for each phase: `received`, `preprocessed`, `model_loaded`, `masks_generated`, `objects_saved`, `completed`.

* Include timings and counts in logs and `metadata.json`.

## Response Format

* `200 OK` JSON:

  * `success`: true,

  * `job_id`: UUID string,

  * `objects_count`: number,

  * `output_dir`: relative path,

  * `files`: { `overlay`, `masks`: \[..], `objects`: \[..], `metadata` },

  * `parameters`: mask generator params used.

* Error JSON includes `success: false`, `error_code`, `message`, and `details`.

## File Changes (≤200 LOC per file)

* `app/routes/models.py`: remove file.

* `app/routes/segment.py`: rewrite to a single endpoint using the pipeline.

* `app/services/pipeline.py`: new file orchestrating the processing.

* `app/services/image_processor.py`: minor fix to `validate_image` (use MB→bytes conversion), plus helper to persist masks.

* `app/services/model_manager.py`: keep, but remove any route exposure; continue to provide one‑time weight download and cached predictor.

* `app/main.py`: remove models router; hook bootstrap in lifespan.

* `requirements.txt`: pin `segment-anything` via GitHub VCS URL.

## Notes on the Notebook Alignment

* The plan mirrors the notebook flow:

  * Dependency from GitHub (library code),

  * `SamAutomaticMaskGenerator` for mask creation,

  * Object extraction with RGBA outputs,

  * Saved artifacts per run with organized naming.

## Testing & Validation

* Add Playwright MCP API tests for `POST /segment`:

  * Upload sample PNG/JPEG; assert `200` and output files exist.

  * Upload non‑image; assert `400 INVALID_IMAGE`.

  * Simulate CUDA OOM (env flag) to verify retry and recovery path.

* Manual verification: run local server, post form‑data image, inspect `results/` directory structure.

## Migration & Backward Compatibility

* Legacy `/segment/*` routes and `/models/*` become unavailable; `/jobs/*` remains untouched unless you prefer removal.

* Docker images unchanged; Celery stack may be left in place but unused by the consolidated endpoint.

## Acceptance Criteria

* Only one segmentation endpoint exists.

* First run downloads weights once, subsequent runs reuse cache; library comes from GitHub.

* End‑to‑end pipeline produces correct masks and RGBA objects, saves files, validates outputs, logs structured status, and returns a clean JSON response.

