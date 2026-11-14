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


N = 2000
A = np.full((N, N), 1.0)
x = np.full(N, 3.0)
y = np.empty(N)          # empty box for the answer, made BEFORE timing

t = time_it(lambda: np.matmul(A, x, out=y))

flops = 2 * N**2
nbytes = (N**2 + 2 * N) * A.itemsize

print("time:", t, "seconds")
print("GFLOP/s:", flops / t / 1e9)
print("GB/s:", nbytes / t / 1e9)
print("check answer:", y[0])