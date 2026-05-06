# LAMP-KV

**LAMP-KV: Memory-Pressure-Aware Mixed-Precision KV-Cache Compression for Memory-Constrained LLM Inference**

Author: John Cheung  
Affiliation: College of Professional and Continued Education

This repository contains the arXiv-style paper source and the supplementary testing program for peer review.

## What this project tests

LAMP-KV does **not** claim to invent KV-cache quantization. Prior work such as KIVI, KVQuant, QAQ, H2O, and SnapKV already validates important KV-cache compression directions.

This project tests a narrower policy question:

> Given a memory-pressure signal, can a runtime choose different KV-cache precisions by block, such as INT8, INT4, INT3, INT2, or FP16 fallback, to obtain a useful memory/error tradeoff?

The current artifact is a CPU reference simulator. It is not an optimized serving kernel.

## Files

```text
paper.tex                         arXiv-style LaTeX source
references.bib                    bibliography
LAMP_KV_arxiv.pdf                 compiled PDF
scripts/lamp_kv_policy_test.py    main testing program
scripts/run_matrix_windows.bat    Windows pressure sweep runner
scripts/summarize_results.py      summarize JSON outputs to CSV
results/README_results.md         pilot result summary
```

## Requirements

Python 3.10+ and NumPy.

Windows example:

```cmd
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe -m pip install numpy
```

If `python` is already in your PATH, you can use `python` instead of the full path.

## Basic test on Windows

From the repository folder:

```cmd
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts\lamp_kv_policy_test.py --batch 1 --seq-len 512 --layers 32 --heads 32 --head-dim 128 --block-tokens 64 --pressure 0.5 --seed 7
```

A larger test used in the paper:

```cmd
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts\lamp_kv_policy_test.py --batch 1 --seq-len 1024 --layers 32 --heads 32 --head-dim 128 --block-tokens 64 --pressure 0.5 --seed 7
```

## Pressure sweep

Run the three main pressure settings:

```cmd
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts\lamp_kv_policy_test.py --pressure 0.0 --seq-len 1024 --seed 7
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts\lamp_kv_policy_test.py --pressure 0.5 --seq-len 1024 --seed 7
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts\lamp_kv_policy_test.py --pressure 1.0 --seq-len 1024 --seed 7
```

Or edit the Python path inside:

```cmd
scripts\run_matrix_windows.bat
```

and run:

```cmd
scripts\run_matrix_windows.bat
```

The batch file writes JSON outputs into `results\`.

## Summarize saved results

```cmd
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts\summarize_results.py results > results\summary.csv
```

## Optional real KV tensor input

The simulator can use real KV tensors saved as an NPZ file with arrays named `k` and `v`:

```cmd
python scripts\lamp_kv_policy_test.py --npz real_kv_sample.npz --head-dim 128 --block-tokens 64 --pressure 0.5
```

Accepted shapes:

```text
[batch, seq_len, layers, heads, head_dim]
[tokens, head_dim]
```

Real KV tensors are required for stronger publication claims. Synthetic tensors are suitable for artifact smoke tests and policy debugging.

## How to interpret the output

Important fields:

- `fp16_kv_bytes`: estimated FP16 KV-cache storage.
- `fixed.int8`, `fixed.int4`, `fixed.int3`, `fixed.int2`: fixed-precision baselines.
- `adaptive`: LAMP-KV policy result.
- `adaptive.bit_histogram`: number of blocks assigned to INT2, INT3, INT4, INT8, or FP16.
- `compression_ratio_vs_fp16`: memory saving relative to FP16.
- `mean_mae`, `p95_mae`, `max_error`: reconstruction-error proxies.
- `timing_ms`: Python reference runtime. Do not treat this as optimized inference latency.

A useful result is not simply the highest compression. A useful adaptive-policy result is one that provides an intermediate tradeoff, for example lower error than fixed INT4 while saving more memory than fixed INT8.

## Pilot result included in the paper

On a Windows ARM notebook using Python 3.13.5 and NumPy 2.4.4, with synthetic KV shape `[1, 1024, 32, 32, 128]`:

| Method | Compression vs FP16 | Mean MAE | Max error |
|---|---:|---:|---:|
| Fixed INT8 | 1.95x | 0.00575 | 0.111 |
| Fixed INT4 | 3.82x | 0.10434 | 2.008 |
| Fixed INT3 | 5.02x | 0.24029 | 4.230 |
| Fixed INT2 | 7.31x | 0.63448 | 10.148 |
| Adaptive p=0.0 | 1.95x | 0.00575 | 0.111 |
| Adaptive p=0.5 | 3.01x | 0.07376 | 1.365 |
| Adaptive p=1.0 | 3.82x | 0.10433 | 1.899 |

## What this artifact can and cannot prove

This artifact can show:

- whether memory pressure changes bit allocation;
- whether adaptive mixed precision differs from fixed-bit baselines;
- memory/error tradeoffs under a reproducible simulator;
- portability on Windows notebooks.

This artifact cannot by itself prove:

- real LLM quality preservation;
- end-to-end LLM serving speedup;
- GPU kernel efficiency;
- DDR5/LPDDR5X bandwidth reduction.

For stronger evaluation, use real KV tensors and integrate the policy into an LLM runtime.

## Recommended GitHub upload

Upload the following to `https://github.com/johncheungmk/LAMP-KV`:

```text
README.md
paper.tex
references.bib
LAMP_KV_arxiv.pdf
scripts/lamp_kv_policy_test.py
scripts/run_matrix_windows.bat
scripts/summarize_results.py
results/README_results.md
```
