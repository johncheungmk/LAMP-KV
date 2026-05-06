# Pilot results

These are the pilot results incorporated into the arXiv-style paper.

Platform reported by the script:

```text
Python: 3.13.5
NumPy: 2.4.4
OS: Windows-11-10.0.26200-SP0
Processor: ARMv8 (64-bit) Family 8 Model 1 Revision 201, Qualcomm Technologies Inc
Notebook context: Lenovo Yoga Slim 7 14Q8X9-class LPDDR5X notebook
```

Synthetic KV configuration:

```text
batch = 1
seq_len = 1024
layers = 32
heads = 32
head_dim = 128
block_tokens = 64
outlier_prob = 0.001
outlier_scale = 6.0
seed = 7
FP16 KV bytes = 536,870,912
```

Fixed-precision results:

| Method | Bytes | Compression vs FP16 | Mean MAE | Max error |
|---|---:|---:|---:|---:|
| INT8 | 274,726,912 | 1.9542x | 0.005752 | 0.111370 |
| INT4 | 140,509,184 | 3.8209x | 0.104336 | 2.008091 |
| INT3 | 106,954,752 | 5.0196x | 0.240294 | 4.230469 |
| INT2 | 73,400,320 | 7.3143x | 0.634478 | 10.148438 |

Adaptive pressure sweep:

| Pressure | Bytes | Compression vs FP16 | Mean MAE | Max error | INT4 blocks | INT8 blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 274,726,912 | 1.9542x | 0.005752 | 0.111370 | 0 | 32,768 |
| 0.5 | 178,085,888 | 3.0147x | 0.073761 | 1.365234 | 23,594 | 9,174 |
| 1.0 | 140,517,376 | 3.8207x | 0.104329 | 1.899414 | 32,766 | 2 |

Interpretation:

- Pressure 0.0 behaves like conservative INT8.
- Pressure 0.5 mixes INT4 and INT8, giving a middle-ground memory/error tradeoff.
- Pressure 1.0 nearly converges to INT4.
- This validates policy behavior on synthetic tensors, not real LLM quality.
