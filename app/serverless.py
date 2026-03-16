"""RunPod serverless handler for MinerU PDF-to-markdown conversion.

Merges optimizations from firecrawl/mineru-api PRs #8, #9, #11, #12, #13:
- GPU thread pinning (cuDNN caches are per-thread)
- Synthetic PDF warmup to trigger torch.compile compilation
- PdfiumError retry with PDF repair
- Specific page extraction
- Processing metadata in response
"""

import base64
import os
import time
import asyncio
import concurrent.futures
import tempfile
import io
import threading

import torch

# Deterministic kernel selection — prevents cuDNN from picking different
# algorithms between warmup and inference.
torch.backends.cudnn.benchmark = False

import runpod

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import ModelSingleton

from pypdf import PdfReader, PdfWriter
from pypdfium2._helpers.misc import PdfiumError

from app.warmup import create_warmup_pdf

# Single-thread GPU executor. cuDNN algorithm caches are per-thread (per
# cuDNN handle), so warmup and real inference MUST run in the same thread
# for compiled kernels to be reused. RunPod serverless processes one job
# at a time per worker anyway.
_gpu_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="gpu"
)


class TimeoutError(Exception):
    pass


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------

def _extract_specific_pages(pdf_bytes: bytes, pages: list) -> bytes:
    """Return a new PDF containing only the specified 1-indexed page numbers."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for page_num in pages:
        idx = page_num - 1
        if 0 <= idx < len(reader.pages):
            writer.add_page(reader.pages[idx])
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _trim_pdf_to_max_pages(pdf_bytes: bytes, max_pages: int) -> bytes:
    """Return a new PDF with at most the first max_pages pages."""
    if max_pages is None or max_pages <= 0:
        return pdf_bytes
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for i in range(min(max_pages, len(reader.pages))):
        writer.add_page(reader.pages[i])
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _repair_pdf(pdf_bytes: bytes) -> bytes:
    """Re-write PDF through pypdf to fix structural issues."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()
    except Exception:
        return pdf_bytes


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def _do_convert(pdf_bytes, lang, parse_method, formula_enable, table_enable,
                max_pages, pages, start_time):
    """Core conversion: doc_analyze -> result_to_middle_json -> union_make.

    Returns (md_content, content_list, metadata).
    """
    if pages is not None:
        pdf_bytes = _extract_specific_pages(pdf_bytes, pages)

    if max_pages is not None:
        pdf_bytes = _trim_pdf_to_max_pages(pdf_bytes, int(max_pages))

    infer_results, all_image_lists, all_pdf_docs, lang_list_result, ocr_enabled_list = pipeline_doc_analyze(
        [pdf_bytes],
        [lang],
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
    )

    model_list = infer_results[0]
    images_list = all_image_lists[0]
    pdf_doc = all_pdf_docs[0]
    page_count = len(pdf_doc)
    _lang = lang_list_result[0]
    _ocr_enable = ocr_enabled_list[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        image_writer = FileBasedDataWriter(temp_dir)
        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_doc, image_writer,
            _lang, _ocr_enable, formula_enable,
        )
        pdf_info = middle_json["pdf_info"]
        md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, "images")
        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, "images")

    processing_time_ms = round((time.time() - start_time) * 1000)
    metadata = {
        "pages": page_count,
        "ocr": _ocr_enable,
        "processing_time_ms": processing_time_ms,
    }
    return md_content, content_list, metadata


def _convert_single(pdf_bytes, start_time, lang="en", parse_method="auto",
                    formula_enable=True, table_enable=True, max_pages=None,
                    pages=None):
    """Convert a single PDF with PdfiumError retry logic."""
    try:
        return _do_convert(pdf_bytes, lang, parse_method, formula_enable,
                           table_enable, max_pages, pages, start_time)
    except PdfiumError as first_error:
        repaired = _repair_pdf(pdf_bytes)
        if repaired is pdf_bytes:
            raise Exception(f"Error converting PDF to markdown: {first_error}")
        try:
            return _do_convert(repaired, lang, parse_method, formula_enable,
                               table_enable, max_pages, pages, start_time)
        except Exception:
            raise Exception(f"Error converting PDF to markdown: {first_error}")
    except Exception as e:
        raise Exception(f"Error converting PDF to markdown: {e}")


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

async def async_convert_to_markdown(pdf_bytes, timeout_seconds=None, **kwargs):
    """Async wrapper running conversion on the GPU executor with timeout."""
    loop = asyncio.get_running_loop()
    start_time = time.time()

    async def _run():
        return await loop.run_in_executor(
            _gpu_executor,
            lambda: _convert_single(pdf_bytes, start_time, **kwargs),
        )

    if timeout_seconds and timeout_seconds > 0:
        try:
            return await asyncio.wait_for(_run(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"PDF processing timed out after {timeout_seconds} seconds")
    else:
        return await _run()


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

async def handler(event):
    """Main serverless handler."""
    try:
        input_data = event.get("input", {})
        base64_content = input_data.get("file_content")
        filename = input_data.get("filename")
        timeout = input_data.get("timeout")
        created_at = input_data.get("created_at")
        max_pages = input_data.get("max_pages")
        pages = input_data.get("pages")

        lang = input_data.get("lang", "en")
        # Default to "ocr" when specific pages are requested (caller already
        # determined these pages need OCR)
        parse_method = input_data.get("parse_method", "ocr" if pages else "auto")
        formula_enable = input_data.get("formula_enable", True)
        table_enable = input_data.get("table_enable", True)

        # Calculate remaining timeout
        timeout_seconds = None
        if timeout:
            timeout_seconds = int(timeout) / 1000
            if created_at:
                elapsed = time.time() - (created_at / 1000)
                if elapsed >= timeout_seconds:
                    return {"error": "Request timed out before processing", "status": "TIMEOUT"}
                timeout_seconds = max(0, timeout_seconds - elapsed)
                if timeout_seconds < 1:
                    return {"error": "Insufficient time remaining", "status": "TIMEOUT"}

        # Validate input
        if not base64_content or not filename:
            return {"error": "Missing file_content or filename", "status": "ERROR"}

        if not filename.lower().endswith('.pdf'):
            return {"error": "Only PDF files supported", "status": "ERROR"}

        if max_pages is not None:
            try:
                max_pages = int(max_pages)
                if max_pages <= 0:
                    return {"error": "max_pages must be a positive integer", "status": "ERROR"}
            except Exception:
                return {"error": "Invalid max_pages; must be an integer", "status": "ERROR"}

        if pages is not None:
            if not isinstance(pages, list):
                return {"error": "pages must be a list of positive integers", "status": "ERROR"}
            try:
                pages = [int(p) for p in pages]
                if not all(p > 0 for p in pages):
                    return {"error": "pages must contain only positive integers", "status": "ERROR"}
            except Exception:
                return {"error": "Invalid pages; must be a list of integers", "status": "ERROR"}

        pdf_bytes = base64.b64decode(base64_content)

        md_content, content_list, metadata = await async_convert_to_markdown(
            pdf_bytes=pdf_bytes,
            timeout_seconds=timeout_seconds,
            lang=lang,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            max_pages=max_pages,
            pages=pages,
        )

        return {
            "markdown": md_content,
            "content_list": content_list,
            "status": "SUCCESS",
            **metadata,
        }

    except TimeoutError as e:
        return {"error": str(e), "status": "TIMEOUT"}
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup_with_inference():
    """Run a full inference pass on a synthetic PDF to trigger torch.compile.

    2 pages is sufficient — torch.compile(dynamic=True) compiles once for
    arbitrary shapes, so extra pages are redundant inference cycles.
    """
    thread = threading.current_thread().name
    print(f"[{thread}] Running warm-up inference to trigger torch.compile...")
    start = time.time()
    pdf_bytes = create_warmup_pdf(num_pages=2)
    try:
        _do_convert(
            pdf_bytes, lang="en", parse_method="ocr",
            formula_enable=True, table_enable=True,
            max_pages=None, pages=None, start_time=time.time(),
        )
        elapsed = round(time.time() - start, 1)
        print(f"[{thread}] Warm-up inference complete in {elapsed}s")
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"[{thread}] Warm-up inference finished in {elapsed}s (non-critical error: {e})")


def _full_warmup():
    """Load models + run synthetic inference on the GPU thread.

    Executed inside _gpu_executor so cuDNN algorithm caches live in the
    same thread that will later handle real inference requests.
    """
    warmup_start = time.time()
    thread = threading.current_thread().name

    # Phase 1: Model loading
    print(f"[{thread}] Loading pipeline models...")
    t0 = time.time()
    ModelSingleton().get_model(
        lang="en",
        formula_enable=True,
        table_enable=True,
    )
    model_load_s = round(time.time() - t0, 1)
    print(f"[{thread}] Pipeline models loaded in {model_load_s}s")

    # Phase 2: Warmup inference (triggers torch.compile)
    t0 = time.time()
    _warmup_with_inference()
    inference_s = round(time.time() - t0, 1)

    total_s = round(time.time() - warmup_start, 1)
    print(f"[{thread}] Full warmup complete in {total_s}s "
          f"(model_load={model_load_s}s, inference={inference_s}s)")


def _setup_torch_cache():
    """Namespace TorchInductor cache by GPU compute capability.

    Triton kernels are architecture-specific — an A100 (sm_80) cache can't
    be used on an L4 (sm_89). Appending the compute capability prevents
    collisions when TORCHINDUCTOR_CACHE_DIR points to a shared persistent
    volume (e.g. /runpod-volume/torch_cache).

    With a persistent volume, the first cold start compiles and populates
    the cache (~60-120s). Subsequent starts on the same GPU type load
    cached kernels (~1-2s) — warmup inference still runs for CUDA graph
    capture but skips the expensive Triton compilation.
    """
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    gpu_arch = f"sm_{major}{minor}"
    base_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/app/.torch_cache")
    cache_dir = os.path.join(base_dir, gpu_arch)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    try:
        has_entries = bool(os.listdir(cache_dir))
    except OSError:
        has_entries = False
    state = "pre-populated" if has_entries else "empty"
    print(f"  torch_cache: {cache_dir} ({state})")


def _gpu_diagnostics():
    """Print GPU/CUDA diagnostics at startup."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        mem = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(mem / (1024**3), 1)
    try:
        from mineru.utils.config_reader import get_device
        info["mineru_device"] = get_device()
    except Exception:
        pass
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== GPU Diagnostics ===")
    _diag = _gpu_diagnostics()
    _setup_torch_cache()
    print("=======================")

    # Run all warmup on the GPU thread so cuDNN caches are reused at inference time
    _gpu_executor.submit(_full_warmup).result()

    if os.environ.get("DEBUG_SERVER", "false").lower() == "true":
        import uvicorn
        from fastapi import FastAPI, Request

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/debug")
        async def debug_info():
            return _diag

        @app.post("/run")
        async def debug_endpoint(request: Request):
            input_data = await request.json()
            event = {"input": input_data}
            return await handler(event)

        print("Starting Debug Server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
