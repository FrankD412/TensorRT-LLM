# GMS Model Loader Integration for TensorRT-LLM

**Date:** 2026-03-18
**Status:** Approved

## Overview

Integrate the `gpu_memory_service` (GMS) monkey-patch into TensorRT-LLM's model loading pipeline. When enabled, each MPI subprocess patches `ModelLoader.load` and `get_rank_model_storage` to use GMS-backed weight storage, enabling weight sharing across inference processes.

## Motivation

The `gpu_memory_service` package (from the dynamo project) provides a patched model loader that can load weights into GMS-managed GPU memory mappings. TRT-LLM spawns MPI subprocesses for multi-GPU inference via `mpi4py.futures.MPIPoolExecutor`. Each subprocess independently loads model weights (each rank loads its own shard), so the patch must be applied in every rank before `ModelLoader.load()` runs.

## Design

### Single Change

**File:** `tensorrt_llm/executor/worker.py`
**Function:** `worker_main()`

The patch is injected **inside the existing `try` block** (currently starting at line 280), immediately before `worker_cls(...)` is constructed. Placing it here reuses the existing exception handler: any failure (e.g., missing package) is logged on all ranks, and rank 0 (the leader) forwards the error to the proxy via `worker_init_status_queue`, which surfaces it to the user. Non-leader ranks only log locally and return — the user-visible error comes from rank 0.

```python
try:
    # GMS patch injection — runs on all MPI ranks before model loading
    if os.environ.get("TRTLLM_GMS_ENABLED", "").lower() in ("1", "true"):
        try:
            from gpu_memory_service.integrations.trtllm.model_loader import (
                patch_model_loader,
                set_gms_enabled,
            )
        except ImportError as e:
            raise ImportError(
                "TRTLLM_GMS_ENABLED is set but gpu_memory_service is not installed. "
                "Install it or unset TRTLLM_GMS_ENABLED."
            ) from e
        patch_model_loader()
        set_gms_enabled(True)

    worker: GenerationExecutorWorker = worker_cls(engine, ...)

# NOTE: The actual except block below is the pre-existing handler in worker_main().
# The `if is_leader:` guard is already there — worker_init_status_queue only
# exists on rank 0. This pseudocode is abbreviated.
except Exception as e:
    logger.error(...)          # logged on all ranks
    if is_leader:              # only rank 0 forwards the error to the proxy
        worker_init_status_queue.notify_with_retry((e, traceback.format_exc()))
    return
```

### All MPI Ranks Apply the Patch

The patch block runs on **all ranks**, not just rank 0. This is intentional: each rank calls `ModelLoader.load()` independently to load its weight shard, so each rank must have the patched version. `patch_model_loader()` is idempotent (guarded by `_model_loader_patched`). `set_gms_enabled(True)` sets a module-level boolean and is safe to call multiple times.

If the patch raises on a non-leader rank (e.g., package missing), that rank logs the error and returns. The user-visible error is delivered only by rank 0. In practice all ranks share the same Python environment, so a missing package will cause rank 0 to raise as well.

### Opt-In Mechanism

- **Environment variable:** `TRTLLM_GMS_ENABLED=1` (also accepts `true`, case-insensitive)
- Env vars set before the MPI pool is initialized propagate naturally to subprocesses.
- Alternatively, `TRTLLM_GMS_ENABLED` can be passed via `llm_args.env_overrides`. This dict is applied in `worker_main()` before the try block (at line 175, before the second `mpi_comm().barrier()` at line 277 that precedes the patch injection point). This is the recommended path if the env var is set after `import tensorrt_llm`, since MPI caches the environment at import time and post-import env changes may not reach spawned workers.

### What the Patch Does

`patch_model_loader()` (from `gpu_memory_service.integrations.trtllm.model_loader`) monkey-patches two symbols in `tensorrt_llm._torch.pyexecutor.model_loader`:

- `ModelLoader.load` → replaced with a GMS-aware load. In **write mode** (first loader to acquire the lock), weights are allocated into GMS memory mappings and published for sharing. In **read-only mode** (subsequent loaders), weights are imported from an existing GMS mapping instead of loading from disk.
- `get_rank_model_storage` → replaced to return GMS-tracked bytes when available.

`set_gms_enabled(True)` activates the patched code path inside the replaced `ModelLoader.load`. Without this call, the patch is installed but the original load behavior is preserved.

### Lock Mode

The GMS client requests `RW_OR_RO` mode by default: it attempts write mode (exclusive lock to publish weights) and falls back to read-only if another process already holds the write lock. If lock acquisition fails at runtime (e.g., GMS service not running), the resulting exception propagates through the existing `except Exception` handler in `worker_main()` and surfaces via the normal error path. Not configurable in this iteration.

### Backend Scope

The patched symbols (`ModelLoader.load`, `get_rank_model_storage`) live in `tensorrt_llm._torch.pyexecutor.model_loader`. This module is invoked only by the PyTorch backend. The TensorRT backend (`TrtLlmArgs`) builds a TensorRT engine at build time and does not call `ModelLoader.load` at runtime — verified by tracing the TRT backend code path in `tensorrt_llm/executor/`. The patch is therefore installed but never triggered under the TRT backend — harmless.

### Dependency

`gpu_memory_service` is an **optional** external dependency. It is only imported when `TRTLLM_GMS_ENABLED` is set. Errors from a missing package are caught by the nested `try/except ImportError` and re-raised with a clear message, then handled by the existing `worker_main()` error-reporting path.

## Files Changed

| File | Change |
|------|--------|
| `tensorrt_llm/executor/worker.py` | Add GMS patch injection inside the `try` block of `worker_main()` |

## Out of Scope

- Unit tests (deferred)
- `TorchLlmArgs` / Python API flag for GMS (deferred)
- Configurable lock mode via env var (deferred)
