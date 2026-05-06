"""Summarize LAMP-KV JSON results into CSV.

Usage:
  python scripts/summarize_results.py results > results/summary.csv
  python scripts/summarize_results.py              > results/summary.csv
"""
import glob
import json
import os
import sys

result_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
paths = sorted(glob.glob(os.path.join(result_dir, "*.json")))

print("file,pressure,seq_len,fp16_bytes,adaptive_bytes,adaptive_ratio,adaptive_bits,int8_ratio,int4_ratio,int3_ratio,int2_ratio,adaptive_mean_mae,adaptive_max_error,total_ms")
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    adaptive = d["adaptive"]
    fixed = d["fixed"]
    bits = adaptive["bit_histogram"]
    bits_text = ";".join(f"{k}:{v}" for k, v in bits.items())
    cfg = d.get("config", {})
    print(",".join([
        os.path.basename(path),
        str(cfg.get("pressure", "")),
        str(cfg.get("seq_len", "")),
        str(d["fp16_kv_bytes"]),
        str(adaptive["bytes"]),
        f"{adaptive['compression_ratio_vs_fp16']:.4f}",
        bits_text,
        f"{fixed['int8']['compression_ratio_vs_fp16']:.4f}",
        f"{fixed['int4']['compression_ratio_vs_fp16']:.4f}",
        f"{fixed['int3']['compression_ratio_vs_fp16']:.4f}",
        f"{fixed['int2']['compression_ratio_vs_fp16']:.4f}",
        f"{adaptive['mean_mae']:.6f}",
        f"{adaptive['max_error']:.6f}",
        f"{d['timing_ms']['total']:.2f}",
    ]))
