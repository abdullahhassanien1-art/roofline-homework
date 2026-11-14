import csv
import matplotlib.pyplot as plt
import numpy as np

# ---- 1. Read the results file ----
data = []
with open("results.csv") as f:
    for row in csv.DictReader(f):
        data.append({
            "op": row["op"],
            "dtype": row["dtype"],
            "N": int(row["N"]),
            "time": float(row["time_s"]),
            "gflops": float(row["gflops"]),
            "gbs": float(row["gbs"]),
        })


def pick(op, dtype, key):
    """Return the N values and one column for one operation and precision."""
    rows = [r for r in data if r["op"] == op and r["dtype"] == dtype]
    return [r["N"] for r in rows], [r[key] for r in rows]


combos = [("matvec", "float32"), ("matvec", "float64"),
          ("matmul", "float32"), ("matmul", "float64")]
color = {"float32": "tab:blue", "float64": "tab:orange"}   # colour = precision
marker = {"matvec": "o", "matmul": "s"}                     # shape = operation

# ---- 2. Plot: time vs N ----
plt.figure()
for op, dt in combos:
    n, t = pick(op, dt, "time")
    plt.loglog(n, t, marker=marker[op], color=color[dt], label=f"{op} {dt}")
plt.xlabel("N")
plt.ylabel("time [s]")
plt.title("Execution time vs problem size")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig("time_vs_N.png", dpi=150)

# ---- 3. Plot: GFLOP/s vs N ----
plt.figure()
for op, dt in combos:
    n, g = pick(op, dt, "gflops")
    plt.semilogx(n, g, marker=marker[op], color=color[dt], label=f"{op} {dt}")
plt.xlabel("N")
plt.ylabel("performance [GFLOP/s]")
plt.title("Performance vs problem size")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig("gflops_vs_N.png", dpi=150)

# ---- 4. Plot: GB/s vs N ----
plt.figure()
for op, dt in combos:
    n, b = pick(op, dt, "gbs")
    plt.semilogx(n, b, marker=marker[op], color=color[dt], label=f"{op} {dt}")
plt.xlabel("N")
plt.ylabel("effective bandwidth [GB/s]")
plt.title("Effective bandwidth vs problem size")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig("bandwidth_vs_N.png", dpi=150)

# ---- 5. Roofline ----
# Machine limits taken from our own measurements:
#   bandwidth = matvec GB/s at the largest N (data comes from main memory)
#   peak      = best matmul GFLOP/s for each precision
bandwidth = pick("matvec", "float64", "gbs")[1][-1]
peak = {dt: max(pick("matmul", dt, "gflops")[1]) for dt in ["float32", "float64"]}
size = {"float32": 4, "float64": 8}


def intensity(op, N, s):
    """Theoretical FLOPs / bytes."""
    if op == "matvec":
        return 2 * N**2 / ((N**2 + 2 * N) * s)
    return 2 * N**3 / (3 * N**2 * s)


plt.figure(figsize=(7, 5))
ai = np.logspace(-1, 4, 200)
for dt in ["float32", "float64"]:
    plt.loglog(ai, np.minimum(peak[dt], ai * bandwidth), color=color[dt],
               label=f"roofline {dt} (peak {peak[dt]:.0f} GFLOP/s)")
for op, dt in combos:
    n, g = pick(op, dt, "gflops")
    x = [intensity(op, N, size[dt]) for N in n]
    plt.scatter(x, g, marker=marker[op], color=color[dt], edgecolor="k",
                label=f"{op} {dt}", zorder=3)
plt.xlabel("arithmetic intensity [FLOP/byte]")
plt.ylabel("performance [GFLOP/s]")
plt.title(f"Roofline (bandwidth {bandwidth:.0f} GB/s)")
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=8)
plt.savefig("roofline.png", dpi=150)

print("Saved 4 plots")