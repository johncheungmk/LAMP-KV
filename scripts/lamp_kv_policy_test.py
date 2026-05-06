"""
LAMP-KV policy simulator
========================

This script is a reproducible CPU reference implementation for a candidate
research paper: Latency- and Memory-Pressure-Aware Mixed-Precision KV-cache
compression for memory-constrained LLM inference.

It does not claim to be an optimized serving kernel. It evaluates whether a
runtime policy can choose different KV precisions by block under error and
memory-pressure constraints, compared with fixed INT8/INT4/INT3/INT2 baselines.

Run on Windows:
  python lamp_kv_policy_test.py --batch 1 --seq-len 1024 --layers 32 --heads 32 --head-dim 128 --block-tokens 64 --pressure 0.5

Optional real tensor input:
  Provide --npz path_to_file.npz where the file contains arrays named k and v.
  Expected shape is either [batch, seq_len, layers, heads, head_dim] or
  [tokens, head_dim]. If [tokens, head_dim] is used, both k and v must use the
  same shape.
"""

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List, Tuple

import numpy as np


BITS = [2, 3, 4, 8]


def packed_size_bytes(num_values: int, bits: int) -> int:
    return int((num_values * bits + 7) // 8)


def quant_dequant_symmetric(x: np.ndarray, bits: int, axis: int) -> Tuple[np.ndarray, int, int]:
    """Quantize/dequantize with a symmetric scale along the given axis.

    axis=0 approximates per-channel scaling for keys in a [tokens, head_dim]
    block. axis=1 approximates per-token scaling for values.
    Returns dequantized output, quantized storage bytes, and scale bytes.
    """
    if bits == 16:
        return x.astype(np.float32), int(x.size * 2), 0

    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    max_abs = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = max_abs / max(qmax, 1)
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.round(x / scale)
    q = np.clip(q, qmin, qmax).astype(np.int8)
    y = q.astype(np.float32) * scale

    q_bytes = packed_size_bytes(int(x.size), bits)
    scale_bytes = int(scale.size * 2)  # scale stored as fp16 in the estimate
    return y, q_bytes, scale_bytes


def error_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    diff = np.abs(x.astype(np.float32) - y.astype(np.float32))
    return float(np.mean(diff)), float(np.percentile(diff, 95)), float(np.max(diff))


def token_importance(block: np.ndarray) -> float:
    """Synthetic block importance estimator.

    The simulator has no attention scores. We approximate risk by block energy
    and tail magnitude. Real integration should replace this with attention
    mass, layer sensitivity, or downstream calibration loss.
    """
    x = block.astype(np.float32)
    energy = float(np.mean(x * x))
    tail = float(np.percentile(np.abs(x), 99))
    return energy + 0.1 * tail


def choose_thresholds(base_mae: float, base_max: float, pressure: float, importance: float, median_importance: float) -> Tuple[float, float]:
    """Device-aware pressure rule.

    pressure in [0,1]. Larger pressure means memory is scarce and the policy
    tolerates more error. More important blocks receive stricter thresholds.
    """
    pressure = min(1.0, max(0.0, pressure))
    pressure_factor = 0.75 + 1.25 * pressure
    importance_factor = 1.0
    if median_importance > 0:
        rel = importance / median_importance
        importance_factor = 1.0 / np.sqrt(max(rel, 0.25))
        importance_factor = float(np.clip(importance_factor, 0.6, 1.4))
    return base_mae * pressure_factor * importance_factor, base_max * pressure_factor * importance_factor


def eval_block(block: np.ndarray, bits: int, axis: int) -> Dict[str, float]:
    y, q_bytes, scale_bytes = quant_dequant_symmetric(block.astype(np.float32), bits, axis)
    mae, p95, maxe = error_stats(block, y)
    return {
        "bits": bits,
        "bytes": int(q_bytes + scale_bytes),
        "q_bytes": int(q_bytes),
        "scale_bytes": int(scale_bytes),
        "mae": mae,
        "p95_error": p95,
        "max_error": maxe,
    }


def adaptive_block(block: np.ndarray, axis: int, mae_threshold: float, max_threshold: float) -> Dict[str, float]:
    for bits in [2, 3, 4, 8]:
        r = eval_block(block, bits, axis)
        if r["mae"] <= mae_threshold and r["max_error"] <= max_threshold:
            return r
    return {
        "bits": 16,
        "bytes": int(block.size * 2),
        "q_bytes": int(block.size * 2),
        "scale_bytes": 0,
        "mae": 0.0,
        "p95_error": 0.0,
        "max_error": 0.0,
    }


def flatten_kv(k: np.ndarray, v: np.ndarray, head_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes differ: {k.shape} vs {v.shape}")
    if k.ndim == 2:
        if k.shape[1] != head_dim:
            raise ValueError(f"2D k/v head_dim mismatch: {k.shape[1]} vs --head-dim {head_dim}")
        return k.astype(np.float16), v.astype(np.float16)
    if k.ndim == 5:
        return k.reshape(-1, k.shape[-1]).astype(np.float16), v.reshape(-1, v.shape[-1]).astype(np.float16)
    raise ValueError("k and v must be rank-2 [tokens, head_dim] or rank-5 [batch, seq_len, layers, heads, head_dim]")


def load_or_generate(args) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if args.npz:
        data = np.load(args.npz)
        if "k" not in data or "v" not in data:
            raise ValueError("NPZ file must contain arrays named 'k' and 'v'")
        k2, v2 = flatten_kv(data["k"], data["v"], args.head_dim)
        meta = {"source": "npz", "npz": args.npz, "original_shape": list(data["k"].shape)}
        return k2, v2, meta

    rng = np.random.default_rng(args.seed)
    shape = (args.batch, args.seq_len, args.layers, args.heads, args.head_dim)

    # Synthetic distribution: most entries are Gaussian, with sparse tails.
    k = rng.normal(0, 1.0, size=shape).astype(np.float32)
    v = rng.normal(0, 1.0, size=shape).astype(np.float32)
    if args.outlier_prob > 0:
        k_mask = rng.random(size=shape) < args.outlier_prob
        v_mask = rng.random(size=shape) < args.outlier_prob
        k[k_mask] += rng.normal(0, args.outlier_scale, size=int(k_mask.sum()))
        v[v_mask] += rng.normal(0, args.outlier_scale, size=int(v_mask.sum()))
    meta = {"source": "synthetic", "shape": list(shape), "outlier_prob": args.outlier_prob, "outlier_scale": args.outlier_scale}
    return k.reshape(-1, args.head_dim).astype(np.float16), v.reshape(-1, args.head_dim).astype(np.float16), meta


def summarize_records(records: List[Dict[str, float]], fp16_bytes: int) -> Dict[str, object]:
    total_bytes = int(sum(r["bytes"] for r in records))
    bit_hist = {"2": 0, "3": 0, "4": 0, "8": 0, "16": 0}
    for r in records:
        bit_hist[str(int(r["bits"]))] += 1
    return {
        "bytes": total_bytes,
        "compression_ratio_vs_fp16": float(fp16_bytes / total_bytes) if total_bytes else None,
        "mean_mae": float(np.mean([r["mae"] for r in records])),
        "p95_mae": float(np.percentile([r["mae"] for r in records], 95)),
        "mean_p95_error": float(np.mean([r["p95_error"] for r in records])),
        "max_error": float(np.max([r["max_error"] for r in records])),
        "scale_overhead_bytes": int(sum(r["scale_bytes"] for r in records)),
        "bit_histogram": bit_hist,
    }


def run(args) -> Dict[str, object]:
    t_start = time.perf_counter()
    k2, v2, source_meta = load_or_generate(args)
    load_ms = (time.perf_counter() - t_start) * 1000

    if k2.shape[1] != args.head_dim:
        args.head_dim = int(k2.shape[1])

    block_tokens = args.block_tokens
    k_blocks = [k2[i : i + block_tokens] for i in range(0, k2.shape[0], block_tokens)]
    v_blocks = [v2[i : i + block_tokens] for i in range(0, v2.shape[0], block_tokens)]
    fp16_bytes = int((k2.size + v2.size) * 2)

    importances = [token_importance(b) for b in k_blocks] + [token_importance(b) for b in v_blocks]
    median_importance = float(np.median(importances)) if importances else 1.0

    results = {
        "system": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "config": vars(args),
        "source": source_meta,
        "tokens_total": int(k2.shape[0]),
        "head_dim": int(k2.shape[1]),
        "num_blocks_per_cache": len(k_blocks),
        "fp16_kv_bytes": fp16_bytes,
        "fixed": {},
        "adaptive": {},
        "timing_ms": {"load_or_generate": load_ms},
    }

    t0 = time.perf_counter()
    for bits in [8, 4, 3, 2]:
        recs = []
        for kb, vb in zip(k_blocks, v_blocks):
            recs.append(eval_block(kb, bits, axis=0))  # K: per-channel
            recs.append(eval_block(vb, bits, axis=1))  # V: per-token
        results["fixed"][f"int{bits}"] = summarize_records(recs, fp16_bytes)
    t1 = time.perf_counter()

    adaptive_recs = []
    thresholds_observed = []
    for kb, vb in zip(k_blocks, v_blocks):
        imp_k = token_importance(kb)
        k_mae_t, k_max_t = choose_thresholds(args.base_mae, args.base_max, args.pressure, imp_k, median_importance)
        thresholds_observed.append((k_mae_t, k_max_t))
        adaptive_recs.append(adaptive_block(kb, axis=0, mae_threshold=k_mae_t, max_threshold=k_max_t))

        imp_v = token_importance(vb)
        v_mae_t, v_max_t = choose_thresholds(args.base_mae, args.base_max, args.pressure, imp_v, median_importance)
        thresholds_observed.append((v_mae_t, v_max_t))
        adaptive_recs.append(adaptive_block(vb, axis=1, mae_threshold=v_mae_t, max_threshold=v_max_t))
    t2 = time.perf_counter()

    adaptive_summary = summarize_records(adaptive_recs, fp16_bytes)
    adaptive_summary.update({
        "mean_mae_threshold": float(np.mean([x[0] for x in thresholds_observed])),
        "mean_max_threshold": float(np.mean([x[1] for x in thresholds_observed])),
    })
    results["adaptive"] = adaptive_summary
    results["timing_ms"].update({
        "fixed_policy_eval": (t1 - t0) * 1000,
        "adaptive_policy_eval": (t2 - t1) * 1000,
        "total": (time.perf_counter() - t_start) * 1000,
    })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-tokens", type=int, default=64)
    parser.add_argument("--pressure", type=float, default=0.5, help="Memory pressure in [0,1]; higher means more aggressive compression")
    parser.add_argument("--base-mae", type=float, default=0.08)
    parser.add_argument("--base-max", type=float, default=1.0)
    parser.add_argument("--outlier-prob", type=float, default=0.001)
    parser.add_argument("--outlier-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--npz", type=str, default="", help="Optional real KV tensor file with arrays named k and v")
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
