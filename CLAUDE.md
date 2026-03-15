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
3. Synthetic 8-page PDF warmup at startup (triggers compilation)
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

**Input:** `file_content` (base64 PDF), `filename`, `lang`, `parse_method`, `formula_enable`, `table_enable`, `max_pages`, `pages` (1-indexed list), `timeout` (ms), `created_at` (epoch ms)

**Output:** `markdown`, `status`, `pages`, `ocr`, `processing_time_ms`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINERU_MODEL_SOURCE` | `local` | Use baked-in models |
| `MINERU_PDF_RENDER_THREADS` | `8` | PDF rendering threads |
| `OMP_NUM_THREADS` | `8` | OpenMP threads |
| `TORCH_COMPILE` | `1` | Enable torch.compile for OCR |
| `TORCH_COMPILE_MODE` | `reduce-overhead` | torch.compile mode (CUDA graphs) |
| `RUNPOD_INIT_TIMEOUT` | `300` | Init timeout (seconds) |
| `DEBUG_SERVER` | `false` | Run as FastAPI on port 8000 |
| `OCR_DET_MAX_FORWARD_BATCH` | `1` | Cap OCR-det batch size |
