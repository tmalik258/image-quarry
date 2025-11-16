## Overview
Implement zero‑shot image classification using OpenAI CLIP (ViT‑B/32) to auto‑label source images, masks, and extracted objects. Integrate with existing FastAPI + Celery pipeline, add batch support, export, grouping, and confidence filtering. Simplify heartbeat logs for readability.

## Components
1. Services
- `app/services/clip_classifier.py`: CLIP loader, preprocessing, text prompt generation, embedding + similarity, thresholding, grouping, export, metrics.
- `app/services/metrics.py`: Simple accuracy, top‑k hit rate, confidence summaries (optional ground truth).
2. Routes
- `app/routes/classify.py`: Endpoints for single image, job masks/objects, and batch classification.
3. Tasks
- Extend `app/services/background_tasks.py`: Celery tasks for batch and job‑scoped classification + cleanup/export.
4. Data
- Persist classifications in DB or alongside job outputs (`results/<job_id>/labels.jsonl`, per‑asset `*.json` with label + confidence).

## Model Loading & Preprocessing
- Add dependency: `git+https://github.com/openai/CLIP.git` (official repo) and `torch` already present.
- Loader: `load_clip(model_name='ViT-B/32', device='cuda' if available else 'cpu')` returning `(model, preprocess)`.
- Preprocessing: use CLIP’s transform pipeline; accept JPEG/PNG/WEBP, auto‑resize as required.

## Image Feature Extraction
- `encode_images(images: List[PIL.Image]) -> torch.Tensor` batched through CLIP image encoder.
- For masks/objects: convert saved PNGs to PIL and encode the same way.

## Dynamic Text Prompt Generation
- No user‑provided label list required.
- Build a broad concept bank on‑the‑fly:
  - Base nouns: COCO classes + OpenImages common nouns + curated top‑N WordNet nouns (bundled list, not user‑provided).
  - Optional context expansion templates: `"a photo of a {noun}"`, `"a {noun}"`, `"the {noun}"`.
  - Heuristics: if objects exist, bias candidates by size/color prominence; otherwise use base bank.
- Compute text embeddings for candidate prompts once, cache on disk for reuse.

## Similarity & Label Assignment
- Normalize image/text embeddings, compute cosine similarity.
- Top‑k selection per asset (default k=3), return labels + confidence (softmax over similarities or raw cosine scaled).
- Thresholding: configurable `min_confidence` (e.g., 0.25) to filter low confidence; expose via API.

## Thresholding, Removal, Grouping, Export
- Removal: functions to delete assets (images/masks/objects) and DB rows below threshold.
- Grouping:
  - Primary: group by top‑1 predicted label.
  - Fallback: KMeans in CLIP image embedding space to cluster unlabeled/low‑confidence items; assign cluster names via nearest text embedding.
- Export:
  - `results/<job_id>/labels.jsonl` with `{path, labels:[{name,score}], top1, confidence}`.
  - Optional ZIP export of grouped folders, plus a CSV of `path, top1, confidence`.

## Performance Metrics
- If ground truth provided (optional): accuracy, top‑k accuracy, confusion matrix.
- Otherwise: report similarity distribution stats (mean/std), label diversity, percent above threshold.
- Log per‑batch timing, throughput (items/sec), GPU/CPU mem snapshots.

## Batch Processing
- Celery task `classify_batch_task(images, params)`:
  - Loads CLIP once, encodes in batches (configurable batch size), computes labels, writes outputs.
  - Supports classifying a job’s masks/objects via paths in `results/<job_id>/`.

## API Endpoints
- `POST /classify/image` — body: image + params; returns labels + confidence.
- `POST /classify/job/{job_id}` — classify `source.png`, masks, and objects; accepts threshold + export options.
- `POST /classify/batch` — accept list of images (or job ids) for batch classification.
- `POST /classify/cleanup/{job_id}` — remove assets below threshold.
- `GET /classify/export/{job_id}` — stream ZIP/CSV of labeled dataset.

## Data Flow Integration
- After segmentation completes, optionally run classification task for `source.png`, `masks/`, `objects/` and write `labels.jsonl`.
- `GET /jobs/{job_id}/files` extended to include `labels.jsonl` path.

## Performance & Scalability
- Batch encoding with `torch.no_grad()` and `pin_memory` for CUDA.
- Cache text embeddings to avoid recompute across requests.
- Configurable batch size, mixed precision on CUDA.
- Reuse single CLIP model per worker.

## Logging Simplification
- Replace heartbeat logs with concise, aligned key‑value:
  - Format: `HB op=<id> job=<id|none> dur=<sec>s rss=<MB> gpu=<alloc>/<res>MB`
  - Example: `HB op=fae757d0 job=none dur=40.46s rss=1236MB gpu=2934/6264MB`
- Apply in `app/services/image_processor.py` where heartbeat is logged.

## Error Handling
- Wrap model load/encode in try/except; retry once if CUDA OOM, fallback to CPU.
- Validate input formats; skip unreadable assets and log warnings.
- Robust file IO with existence checks.

## Security & Windows Compatibility
- No secrets in logs; avoid external downloads at runtime.
- Paths are Windows‑safe; use `os.path.join`, avoid shell commands.

## Changes Summary
- New: `app/services/clip_classifier.py`, `app/services/metrics.py`, `app/routes/classify.py`.
- Update: `requirements.txt` add `git+https://github.com/openai/CLIP.git`.
- Update: `app/services/background_tasks.py` to add Celery tasks and optional post‑segmentation classification.
- Update: `app/routes/jobs.py` to expose labeled artifacts (file listing).
- Update: Heartbeat log format in `app/services/image_processor.py`.

## Confirmation
If this plan matches your goals, I will implement the services, endpoints, tasks, dependency update, and logging simplification. Please confirm to proceed.