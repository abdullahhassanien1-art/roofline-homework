import csv
import time
import numpy as np


def time_it(func, repeats=7):
    func()          # warm-up run 1 (not timed)
    func()          # warm-up run 2 (not timed)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)   # fastest run


def measure_matvec(N, dtype):
    A = np.full((N, N), 1.0, dtype=dtype)
    x = np.full(N, 3.0, dtype=dtype)
    y = np.empty(N, dtype=dtype)

    t = time_it(lambda: np.matmul(A, x, out=y))

    flops = 2 * N**2
    nbytes = (N**2 + 2 * N) * A.itemsize
    return t, flops / t / 1e9, nbytes / t / 1e9


def measure_matmul(N, dtype):
    A = np.full((N, N), 1.0, dtype=dtype)
    B = np.full((N, N), 2.0, dtype=dtype)
    C = np.empty((N, N), dtype=dtype)

    t = time_it(lambda: np.matmul(A, B, out=C))

    flops = 2 * N**3
    nbytes = 3 * N**2 * A.itemsize
    return t, flops / t / 1e9, nbytes / t / 1e9


rows = []

for dtype in [np.float32, np.float64]:
    for N in [256, 512, 1024, 2048, 4096, 8192]:
        t, gflops, gbs = measure_matvec(N, dtype)
        rows.append(["matvec", np.dtype(dtype).name, N, t, gflops, gbs])
        print(rows[-1])

for dtype in [np.float32, np.float64]:
    for N in [128, 256, 512, 1024, 2048, 4096]:
        t, gflops, gbs = measure_matmul(N, dtype)
        rows.append(["matmul", np.dtype(dtype).name, N, t, gflops, gbs])
        print(rows[-1])

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["op", "dtype", "N", "time_s", "gflops", "gbs"])
    writer.writerows(rows)

print("Saved results.csv")