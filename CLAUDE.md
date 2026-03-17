# MinerU RunPod Serverless Worker

RunPod serverless worker that runs MinerU for PDF-to-markdown conversion with optimized cold start.

## Architecture

- `app/serverless.py` — RunPod handler, GPU thread pool (max_workers=1), conversion pipeline, debug server
- `app/warmup.py` — Synthetic PDF generator for startup warmup (triggers torch.compile)
- `patch/` — 6 patches applied to MinerU internals at Docker build time
- `Dockerfile` — nvidia/cuda base, uv for deps, patches applied, models baked in

## Cold Start Optimizations

1. Models baked into Docker image (no runtime download)
2. `torch.compile(dynamic=True)` on OCR-det and OCR-rec (compiled kernels handle all shapes)
3. Synthetic 2-page PDF warmup at startup (triggers compilation)
4. GPU thread pinning (cuDNN caches are per-thread)
5. OCR-det batch size capped to N=1 (matches compiled kernel cache)
6. Layout/MFD batch sizes increased to 8
7. Table UNet moved from CPU to GPU
8. `cudnn.benchmark = False` for deterministic kernel selection

## Development

```bash
# Build
docker build -t mineru-runpod:latest .

# Run debug server
docker run --gpus all -e DEBUG_SERVER=true -p 8000:8000 mineru-runpod:latest

# Test
BASE64=$(base64 -w 0 test.pdf)
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d "{\"file_content\": \"$BASE64\", \"filename\": \"test.pdf\"}"
```

## API

**Input (S3 mode):** `s3_key`, `lang`, `parse_method`, `formula_enable`, `table_enable`, `max_pages`, `pages` (1-indexed list), `timeout` (ms), `created_at` (epoch ms)

**Input (base64 mode):** `file_content` (base64 PDF), `filename`, `lang`, `parse_method`, `formula_enable`, `table_enable`, `max_pages`, `pages` (1-indexed list), `timeout` (ms), `created_at` (epoch ms)

**Output:** `markdown`, `content_list`, `status`, `pages`, `ocr`, `processing_time_ms`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_ENDPOINT` | — | SeaweedFS S3 endpoint URL |
| `S3_BUCKET` | — | S3 bucket name |
| `S3_ACCESS_KEY` | — | S3 access key |
| `S3_SECRET_KEY` | — | S3 secret key |
| `S3_REGION` | `us-east-1` | S3 region |
| `MINERU_MODEL_SOURCE` | `local` | Use baked-in models |
| `HF_HUB_OFFLINE` | `1` | Prevent huggingface_hub HTTP calls |
| `TRANSFORMERS_OFFLINE` | `1` | Prevent transformers HTTP calls |
| `CUDA_MODULE_LOADING` | `LAZY` | Defer unused CUDA kernel loading |
| `TORCHINDUCTOR_CACHE_DIR` | `/app/.torch_cache` | torch.compile cache (GPU arch auto-appended). Point to a RunPod network volume to persist across cold starts |
| `MINERU_PDF_RENDER_THREADS` | `8` | PDF rendering threads |
| `OMP_NUM_THREADS` | `8` | OpenMP threads |
| `TORCH_COMPILE` | `1` | Enable torch.compile for OCR |
| `TORCH_COMPILE_MODE` | `reduce-overhead` | torch.compile mode (CUDA graphs) |
| `RUNPOD_INIT_TIMEOUT` | `300` | Init timeout (seconds) |
| `DEBUG_SERVER` | `false` | Run as FastAPI on port 8000 |
| `OCR_DET_MAX_FORWARD_BATCH` | `1` | Cap OCR-det batch size |

## RunPod Secrets

S3 credentials are injected via RunPod secrets at the endpoint level — **one Docker image serves both dev and prod**. Create two serverless endpoints from the same image, each referencing its own set of secrets.

Secrets are not visible to endpoints unless explicitly referenced as env vars using `{{ RUNPOD_SECRET_<name> }}`.

### Secret Names

| Env Variable | Dev Secret | Prod Secret |
|---|---|---|
| `S3_ENDPOINT` | `dev_s3_endpoint` | `prod_s3_endpoint` |
| `S3_BUCKET` | `dev_s3_bucket` | `prod_s3_bucket` |
| `S3_ACCESS_KEY` | `dev_s3_access_key` | `prod_s3_access_key` |
| `S3_SECRET_KEY` | `dev_s3_secret_key` | `prod_s3_secret_key` |
| `S3_REGION` | `dev_s3_region` | `prod_s3_region` |

### Setup

1. Create all 10 secrets in RunPod (Settings > Secrets)
2. Build and push **one** Docker image
3. Create **two** serverless endpoints from the same image:
   - **Dev endpoint** — set env vars referencing `dev_*` secrets (e.g. `S3_ENDPOINT={{ RUNPOD_SECRET_dev_s3_endpoint }}`)
   - **Prod endpoint** — set env vars referencing `prod_*` secrets (e.g. `S3_ENDPOINT={{ RUNPOD_SECRET_prod_s3_endpoint }}`)
