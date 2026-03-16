FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        patch curl g++ libopencv-dev python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
        "mineru[pipeline]>=2.7.0,<3.0.0" \
        "runpod>=1.7.12,<2.0.0" \
        "pypdf>=4.2.0,<6.0.0" && \
    rm -rf /root/.cache/uv /root/.cache/pip

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Apply all 6 patches to MinerU internals

# Batch processing + VRAM-based batch_ratio (32 for 22GB+), force OCR-det batching
COPY patch/mineru_batch.patch /tmp/mineru_batch.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/backend/pipeline/pipeline_analyze.py < /tmp/mineru_batch.patch

# Cap OCR-det forward batch size to N=1 so CUDA kernels match warmup cache
COPY patch/batch_analyze_det_bs.patch /tmp/batch_analyze_det_bs.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/backend/pipeline/batch_analyze.py < /tmp/batch_analyze_det_bs.patch

# Increase Layout/MFD batch sizes from 1 to 8 for better GPU utilization
COPY patch/batch_sizes.patch /tmp/batch_sizes.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/backend/pipeline/batch_analyze.py < /tmp/batch_sizes.patch

# Enable CUDA for wired table UNet model (was CPU-only)
COPY patch/wired_table_cuda.patch /tmp/wired_table_cuda.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/model/table/rec/unet_table/utils.py < /tmp/wired_table_cuda.patch

# torch.compile for OCR detection model
COPY patch/torch_compile_ocr_det.patch /tmp/torch_compile_ocr_det.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/model/utils/tools/infer/predict_det.py < /tmp/torch_compile_ocr_det.patch

# torch.compile for OCR recognition model
COPY patch/torch_compile_ocr_rec.patch /tmp/torch_compile_ocr_rec.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/model/utils/tools/infer/predict_rec.py < /tmp/torch_compile_ocr_rec.patch

# Download models at build time (avoids runtime download of ~5-10GB)
RUN mineru-models-download -s huggingface -m pipeline

# Copy application code
COPY app/ ./app/

# Environment configuration
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCHINDUCTOR_CACHE_DIR=/app/.torch_cache
ENV MINERU_MODEL_SOURCE=local
ENV MINERU_PDF_RENDER_THREADS=8
ENV OMP_NUM_THREADS=8
ENV TORCH_COMPILE=1
ENV TORCH_COMPILE_MODE=reduce-overhead
ENV OCR_DET_MAX_FORWARD_BATCH=1
ENV RUNPOD_INIT_TIMEOUT=300

ENTRYPOINT ["python3", "-m", "app.serverless"]
