# Roofline model homework

Benchmarks matrix–vector (y = Ax) and matrix–matrix (C = AB) multiplication
with NumPy in float32 and float64, and plots the results with the Roofline model.

## Files
- `benchmark.py`: runs both operations for several sizes N and saves `results.csv`
- `plot.py`: reads `results.csv` and makes the 4 plots (`.png` files)
- `results.csv`: my measured results

## How to run
pip3 install numpy matplotlib
python3 benchmark.py
python3 plot.py

## How the timing works
- Arrays are created before timing; results are written into existing arrays with `out=`
- 2 warm-up runs, then 7 timed runs; the fastest is kept
- FLOPs: 2N² (matvec), 2N³ (matmul)
- Bytes: (N² + 2N)·s (matvec), 3N²·s (matmul), where s = 4 (float32) or 8 (float64)
