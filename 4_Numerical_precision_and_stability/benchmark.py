import time
import numpy as np

N = 2000

A = np.full((N, N), 1.0)   # a 2000 × 2000 grid, all 1.0
x = np.full(N, 3.0)        # a list of 2000 numbers, all 3.0

start = time.perf_counter()   # start the stopwatch
y = A @ x                     # matrix × vector
end = time.perf_counter()     # stop the stopwatch

print("time:", end - start, "seconds")
print("first answer:", y[0])