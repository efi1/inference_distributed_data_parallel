## Training throughput

| Metric | 1-GPU | 4-GPU |
|---|---|---|
| `train_runtime` (s) | 196.400 | 2,480 |
| `train_samples_per_second` | 20.370 | 6.452 |
| `train_steps_per_second` | 2.546 | 0.202 |
| `train_loss` (final) | 8.232 | 8.042 |
| Avg step time from `[Comm]` (s) | 0.3624 | 4.9307 |

## Inference (evaluation) throughput

| Metric | 1-GPU | 4-GPU |
|---|---|---|
| `eval_samples_per_second` | 53.671 | 200.649 |
| `eval_runtime` (s) | 4.639 | 1.241 |
| `eval_loss` | 6.990 | 6.801 |
| `eval_steps_per_second` | 53.671 | 50.767 |

## Communication numbers

| Metric | 1-GPU | 4-GPU |
|---|---|---|
| world_size | 1 | 4 |
| Trainable parameters | 774,030,080 | 774,030,080 |
| Theoretical grad payload / step (bytes) | 3,096,120,320 | 3,096,120,320 |
| Theoretical ring all-reduce bytes/GPU/step | 0 | 4,644,180,480 |
| Total measured comm time (s) | 0.0000 | 2,297.7714 |
| Total measured comm bytes | 0 | 1,548,060,160,000 |
| Avg comm time per all-reduce (s) | 0.000000 | 0.042245 |
| Avg comm time per optimizer step (s) | 0.000000 | 4.595543 |

## Derived scaling figures

- Wall-clock speedup (runtime 1-GPU / 4-GPU): **0.08×**
- Throughput speedup (samples/s): **0.32×**
- Scaling efficiency vs ideal 4×: **7.9%**
- Measured comm time as share of 4-GPU runtime: **92.7%**
