"""FP32 MLX matmul for GPU-counter traces: `python3 probes/mlx_gemm.py [N] [iters]`.

Warms up, idles 0.5 s, then runs `iters` evaluated N x N x N products and prints
the median host-side time. MLX does not label its command buffers, so
gpu_counters.py selects these by duration instead (--min-ms).
"""
import statistics
import sys
import time

import mlx.core as mx

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
mx.random.seed(0)
a = mx.random.uniform(shape=(n, n), dtype=mx.float32)
b = mx.random.uniform(shape=(n, n), dtype=mx.float32)
mx.eval(a, b)
for _ in range(3):
    mx.eval(a @ b)
mx.synchronize()
time.sleep(0.5)
times = []
for _ in range(iters):
    t = time.perf_counter()
    c = a @ b
    mx.eval(c)
    mx.synchronize()
    times.append(time.perf_counter() - t)
ms = statistics.median(times) * 1e3
print(f'mlx_fp32: median {ms:.2f} ms, {2 * n**3 / ms / 1e6:.0f} GFLOP/s (host time, {iters} iters)')
