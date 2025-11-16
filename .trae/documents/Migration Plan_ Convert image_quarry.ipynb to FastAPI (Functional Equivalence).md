## Overview

* Preserve notebook behavior while adding Celery for model processing and integrating Google Drive for saving outputs.

* Core parity points: SAM automatic and box-prompt segmentation, mask metadata, object extraction with RGBA, visualization alternatives, GPU/weights validation.

## Architecture

* FastAPI application with Celery workers for long-running model tasks.

* Modules:

  * `app/main.py` — entrypoint; lifespan startup checks (weights, device, temp dirs)

  * `app/routes/health.py` — GPU/model health

  * `app/routes/models.py` — weights/device info

  * `app/routes/segment.py` — sync endpoints (auto, box) and async Celery submit endpoints

  * `app/routes/jobs.py` — job/task status and results via Celery backend

  * `app/routes/storage.py` — Google Drive configuration and upload helpers

  * `app/api_schema/` — request/response Pydantic models

  * `app/services/sam_loader.py` — global SAM model lifecycle

  * `app/services/segmentation.py` — image preprocessing, auto/box segmentation

  * `app/services/object_extractor.py` — RGBA object saving and filtering

  * `app/services/storage/google_drive.py` — Google Drive client (OAuth or service account)

  * `app/celery_app.py` — Celery configuration (broker/backend) and task registration

  * `app/config.py` — environment-driven settings (checkpoint path, device, Celery, Drive)

* Keep jobs and background processing via Celery; remove any DB requirements not needed for notebook parity (status stored in Celery backend). If a DB is already present, limit usage to job metadata only.

## Celery Integration

* Broker/backend: Redis recommended (`redis://localhost:6379/0`) or `rpc://` for minimal setup.

* Windows settings: use `CELERY_POOL=solo` for worker.

* Tasks:

  * `tasks.segment_auto(image_bytes, params)` → returns list of mask dicts (notebook-equivalent keys)

  * `tasks.segment_box(image_bytes, box, params)` → returns masks/scores/logits

  * `tasks.extract_objects(image_bytes, params, save_target)` → returns saved paths and counts

* Worker model lifecycle:

  * On worker start, eager-load SAM on `DEVICE` identical to notebook (`cuda:0` if available else `cpu`).

  * Reuse loaded model across tasks to avoid repeated load overhead.

## Google Drive Integration

* Replace Colab `drive.mount` with Drive API-based upload.

* Auth modes:

  * OAuth client ID (user consent flow) for personal Drive

  * Service account for team/shared drive, configured with folder permissions

* Configurable `DRIVE_FOLDER_ID`; optional `DRIVE_PARENT_PATH` mapping

* Endpoints:

  * `POST /storage/drive/config` — upload credentials securely; returns masked config status

  * `POST /storage/drive/upload` — upload a local file path or raw bytes to a specific `folder_id`; returns file id and web view link

* Segmentation endpoints support `save_to_drive:bool` and `drive_folder_id:str` to automatically upload outputs after local save

## Endpoints

* `GET /health` — `{ gpu_available, device, weights_present, checkpoint_path, writable_tmp }`

* `GET /models/checkpoint` — `{ exists, path, size_bytes, sha256 }`

* Sync:

  * `POST /segment/auto` — thresholds identical to notebook; returns list of `MaskItem`

  * `POST /segment/box` — bounding box `{x,y,width,height}` with default fallback; returns masks/scores/logits

* Async (Celery):

  * `POST /segment/auto/async` — queues `segment_auto`; returns `{ task_id }`

  * `POST /segment/box/async` — queues `segment_box`; returns `{ task_id }`

  * `POST /objects/extract/async` — queues `extract_objects`; returns `{ task_id }`

  * `GET /jobs/{task_id}` — status/result; when ready, returns same schema as sync endpoints

## Pydantic Models

* `BoxModel`: `{ x:int, y:int, width:int, height:int }`

* `AutoParamsModel`: `{ points_per_side:int=64, pred_iou_thresh:float=0.90, stability_score_thresh:float=0.92, crop_n_layers:int=1, crop_n_points_downscale_factor:int=2, min_mask_region_area:int=1000 }`

* `MaskItem`: `{ segmentation: (PNG base64|RLE), area:int, bbox:[int,int,int,int], predicted_iou:float, point_coords:[[float,float]], stability_score:float, crop_box:[int,int,int,int] }`

* `AutoSegmentResponse`: `List[MaskItem]`

* `BoxSegmentResponse`: `{ masks: List[MaskItem], scores: List[float], logits: Optional[Any], best: Optional[MaskItem] }`

* `ObjectExtractResponse`: `{ saved: List[str], skipped:int, drive_files: Optional[List[{file_id:str, web_view:str}]] }`

* `JobStatusResponse`: `{ task_id:str, status:"PENDING"|"STARTED"|"SUCCESS"|"FAILURE", result:Optional[Any], error:Optional[str] }`

## Startup & Model Verification

* Validate `SAM_CHECKPOINT_PATH` (existence, readable, optional hash).

* Set `DEVICE` per notebook: `cuda:0` if available else `cpu`.

* Eager-load SAM model; log load duration and memory stats.

* Validate Drive config if `save_to_drive` workflows are enabled.

## Error Handling

* Input validation mirrors notebook (image types, bounding box ranges, threshold ranges).

* Model errors: missing/invalid weights, unsupported `MODEL_TYPE` → `ModelError`.

* Drive errors: invalid credentials, missing `folder_id` → `StorageError`.

* Consistent JSON error structure (existing handlers in `app/main.py:96-165`).

## Functional Validation

* Equivalence tests (pytest):

  * Direct library baseline replicates notebook steps for a deterministic image and the default thresholds.

  * Compare `/segment/auto` sync response with baseline: mask count, top-1 area/bbox, `predicted_iou` tolerance.

  * Compare `/segment/box` with notebook `default_box`.

  * For async: submit tasks, wait for completion, compare results to baseline.

* Performance benchmarks (`pytest-benchmark`): API sync vs Celery worker vs direct library on both CPU and GPU.

* Drive upload tests: use mocked Drive client; verify metadata and count.

## Documentation

* Setup:

  * Python ≥ 3.12; install Torch with appropriate CUDA wheels.

  * `pip install fastapi uvicorn segment-anything supervision opencv-python pillow pydantic celery redis google-api-python-client google-auth google-auth-oauthlib`

  * Configure environment: `SAM_CHECKPOINT_PATH`, `MODEL_TYPE`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `DRIVE_FOLDER_ID`.

* Usage examples:

  * Sync auto: `curl -F image=@image.png "http://localhost:8000/segment/auto?points_per_side=64&min_mask_region_area=1000"`

  * Async auto: `curl -F image=@image.png http://localhost:8000/segment/auto/async` → `GET /jobs/{task_id}`

  * Box: `curl -F image=@image.png -F box='{"x":68,"y":247,"width":555,"height":678}' http://localhost:8000/segment/box`

  * Extract (async) with Drive: `curl -F image=@image.png -F min_pixels=5000 -F save_to_drive=true -F drive_folder_id=YOUR_FOLDER http://localhost:8000/objects/extract/async`

* Migration notes:

  * Colab `drive.mount` replaced by Drive API upload; outputs can still be visible in Drive folders.

  * Interactive `BBoxWidget` replaced by API `box` param with notebook default fallback.

## Troubleshooting

* Weights missing → fix `SAM_CHECKPOINT_PATH`; validate via `/models/checkpoint`.

* GPU off → run on CPU; tune `points_per_side`.

* Celery worker not starting on Windows → ensure `CELERY_POOL=solo`.

* Redis connectivity → check `CELERY_BROKER_URL`/`RESULT_BACKEND`.

* Drive errors → validate credentials and `folder_id` permissions.

## Quality Assurance

* Structured logging on API and worker sides (load times, task durations, errors).

* Health endpoints include GPU/device, weights, broker/backend connectivity.

* Input validation and error codes consistent across sync/async paths.

* Known differences: no Jupyter UI; Drive via API; jobs tracked via Celery backend.

## Codebase References

* Current DB/job-centric segmentation router to be adapted for Celery: `app/routes/segment.py:27-112`, `125-170`, `172-345`.

* DB init in `app/main.py:28-34`; keep only if used for job metadata, else remove.

