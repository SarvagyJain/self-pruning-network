"""Benchmark dense masked inference against CSR sparse inference.

Run this script from the repository root after creating checkpoints with
src/run_pruning_experiment.py. Conversion is timed separately from inference.
Reported storage includes CSR indices and row pointers, not just nonzero values.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from aqua_nn.checkpoint import load_model_checkpoint
from aqua_nn.data.dataset import load_dataset_splits


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"


def current_rss_bytes():
    """Return current process RSS bytes where the OS exposes it."""
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        pass

    if sys.platform.startswith("linux") and Path("/proc/self/statm").exists():
        page_size = os.sysconf("SC_PAGE_SIZE")
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * page_size

    if sys.platform == "win32":
        # Windows working-set measurement without a third-party dependency.
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            process = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            if ok:
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError, TypeError):
            return None

    # macOS / other fallback: resource reports peak RSS, not current RSS.
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, AttributeError):
        return None


def traced_peak_bytes(function, x):
    """Measure Python/NumPy allocations traced during one inference call."""
    gc.collect()
    tracemalloc.start()
    try:
        function(x)
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    finally:
        tracemalloc.stop()


def latency_ms(function, x, warmup, repeats):
    for _ in range(warmup):
        function(x)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(x)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples)), float(np.percentile(samples, 90))


def sparse_matrix_storage_bytes(matrix):
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def batch_from_validation(X, batch_size):
    if len(X) == 0:
        raise ValueError("Validation split is empty")
    if len(X) >= batch_size:
        return np.asarray(X[:batch_size], dtype=np.float64)
    repeats = (batch_size + len(X) - 1) // len(X)
    return np.asarray(np.tile(X, (repeats, 1))[:batch_size], dtype=np.float64)


def benchmark_checkpoint(checkpoint_path, batch_sizes, warmup, repeats):
    model, metadata = load_model_checkpoint(checkpoint_path)
    dataset_name = metadata.get("dataset", "digits")
    seed = int(metadata.get("split_seed", metadata.get("seed", 42)))
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_splits(
        dataset=dataset_name,
        seed=seed,
        validation_fraction=float(metadata.get("validation_fraction", 0.1)),
        test_fraction=float(metadata.get("test_fraction", 0.1)),
    )
    if X_val.shape[1] != model.layer1.weight.data.shape[0]:
        raise ValueError(
            f"Checkpoint expects {model.layer1.weight.data.shape[0]} features, "
            f"but dataset '{dataset_name}' has {X_val.shape[1]}"
        )

    dense_layers = (model.layer1, model.layer2, model.layer3)
    # The checkpoint already stores pruned weights as exact zeros, so reuse those
    # dense arrays rather than multiplying by masks inside every timed call.
    dense_weights = [layer.weight.data for layer in dense_layers]
    dense_weight_bytes = sum(layer.weight.data.nbytes for layer in dense_layers)
    dense_parameter_bytes = sum(parameter.data.nbytes for parameter in model.parameters())
    dense_mask_bytes = sum(layer.mask.nbytes for layer in dense_layers)
    rss_before_sparse = current_rss_bytes()
    conversion_start = time.perf_counter()
    sparse_weights = model.to_sparse_weights()
    conversion_ms = (time.perf_counter() - conversion_start) * 1000.0
    gc.collect()
    rss_after_sparse = current_rss_bytes()
    rss_delta = (
        int(rss_after_sparse - rss_before_sparse)
        if rss_before_sparse is not None and rss_after_sparse is not None
        else None
    )

    csr_weight_bytes = sum(sparse_matrix_storage_bytes(w) for w in sparse_weights)
    bias_bytes = sum(layer.bias.data.nbytes for layer in dense_layers)
    csr_total_with_bias_bytes = csr_weight_bytes + bias_bytes
    csr_components = [
        {
            "shape": list(w.shape),
            "nnz": int(w.nnz),
            "data_bytes": int(w.data.nbytes),
            "indices_bytes": int(w.indices.nbytes),
            "indptr_bytes": int(w.indptr.nbytes),
            "total_csr_bytes": sparse_matrix_storage_bytes(w),
        }
        for w in sparse_weights
    ]

    rows = []
    for batch_size in batch_sizes:
        x = batch_from_validation(X_val, batch_size)
        dense_call = lambda batch: model.predict_logits_dense(batch, dense_weights=dense_weights)
        dense_logits = dense_call(x)
        sparse_logits = model.predict_logits_sparse(x, sparse_weights)
        np.testing.assert_allclose(sparse_logits, dense_logits, rtol=1e-9, atol=1e-9)
        predictions_match = bool(
            np.array_equal(np.argmax(sparse_logits, axis=1), np.argmax(dense_logits, axis=1))
        )

        dense_median, dense_p90 = latency_ms(
            dense_call, x, warmup, repeats
        )
        sparse_median, sparse_p90 = latency_ms(
            lambda batch: model.predict_logits_sparse(batch, sparse_weights),
            x, warmup, repeats,
        )
        dense_peak = traced_peak_bytes(dense_call, x)
        sparse_peak = traced_peak_bytes(
            lambda batch: model.predict_logits_sparse(batch, sparse_weights), x
        )
        try:
            checkpoint_label = str(checkpoint_path.resolve().relative_to(ROOT))
        except ValueError:
            checkpoint_label = str(checkpoint_path.resolve())
        rows.append({
            "checkpoint": checkpoint_label,
            "dataset": dataset_name,
            "seed": seed,
            "target_sparsity": float(metadata.get("target_sparsity", 0.0)),
            "achieved_sparsity": float(
                1.0 - sum(int(np.count_nonzero(layer.mask)) for layer in dense_layers)
                / sum(layer.mask.size for layer in dense_layers)
            ),
            "batch_size": int(batch_size),
            "warmup_runs": int(warmup),
            "timed_repeats": int(repeats),
            "sparse_conversion_ms": float(conversion_ms),
            "dense_median_latency_ms": dense_median,
            "sparse_median_latency_ms": sparse_median,
            "dense_p90_latency_ms": dense_p90,
            "sparse_p90_latency_ms": sparse_p90,
            "median_latency_speedup_dense_over_sparse": (
                float(dense_median / sparse_median) if sparse_median > 0 else None
            ),
            "dense_traced_peak_bytes": dense_peak,
            "sparse_traced_peak_bytes": sparse_peak,
            "dense_weight_storage_bytes": int(dense_weight_bytes),
            "dense_mask_storage_bytes_if_retained": int(dense_mask_bytes),
            "dense_parameter_storage_bytes_including_biases": int(dense_parameter_bytes),
            "csr_weight_storage_bytes_including_indices": int(csr_weight_bytes),
            "csr_parameter_storage_bytes_including_biases": int(csr_total_with_bias_bytes),
            "estimated_weight_storage_reduction_fraction": float(
                1.0 - csr_weight_bytes / dense_weight_bytes
            ) if dense_weight_bytes else None,
            "process_rss_delta_after_csr_conversion_bytes": rss_delta,
            "logits_match_allclose": True,
            "predictions_match_exactly": predictions_match,
            "csr_layers": json.dumps(csr_components),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, action="append",
        help="Checkpoint to benchmark. Repeat to compare multiple checkpoints.",
    )
    parser.add_argument("--batch-sizes", default="1,64,256")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, default=RAW_DIR / "sparse_inference_benchmark.csv")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    try:
        batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(",") if value.strip()]
    except ValueError as exc:
        parser.error(f"Invalid --batch-sizes value: {exc}")
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        parser.error("Every batch size must be a positive integer")

    checkpoints = args.checkpoint or sorted((RAW_DIR / "checkpoints").glob("*.npz"))
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints found. Run src/run_pruning_experiment.py first, "
            "or pass --checkpoint path/to/model.npz."
        )

    rows = []
    for path in checkpoints:
        path = path.resolve()
        print(f"Benchmarking {path}")
        rows.extend(
            benchmark_checkpoint(path, batch_sizes, args.warmup, args.repeats)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "benchmark_rows": len(rows),
        "checkpoints": [str(path) for path in checkpoints],
        "batch_sizes": batch_sizes,
        "warmup_runs": args.warmup,
        "timed_repeats": args.repeats,
        "output": str(args.output),
        "note": (
            "RSS delta is a process-level snapshot and may be noisy; traced peaks "
            "measure allocations during an inference call. Neither metric is a "
            "substitute for repeated isolated-process memory profiling."
        ),
        "results": rows,
    }
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"CSV: {args.output}")
    print(f"JSON: {json_path}")
    for row in rows:
        print(
            f"target={row['target_sparsity']:.0%} sparsity={row['achieved_sparsity']:.2%} "
            f"batch={row['batch_size']} dense={row['dense_median_latency_ms']:.4f}ms "
            f"CSR={row['sparse_median_latency_ms']:.4f}ms "
            f"speedup={row['median_latency_speedup_dense_over_sparse']:.3f}x"
        )


if __name__ == "__main__":
    main()
