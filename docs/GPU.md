# GPU Acceleration — OrbitGuard

OrbitGuard supports optional GPU acceleration for numerically intensive
operations via [CuPy](https://cupy.dev/).  When a CUDA-capable GPU is
available the following computations run on the GPU:

| Operation | Module | Speedup (est.) |
|-----------|--------|----------------|
| Monte Carlo collision probability sampling | `pc_calculator.py` | 10–50× at N ≥ 10 000 |
| Batch distance / norm computation | `screening_engine.py` | 2–5× |
| TCA vectorized norms & dot products | `tca_finder.py` | 2–5× |

Everything else (ODE propagation, Foster/Chan Pc, root-finding, SGP4)
remains on the CPU where it performs best.

---

## Installation

### 1. CUDA toolkit

Install the NVIDIA CUDA toolkit matching your driver:

```bash
# Check driver version
nvidia-smi
```

### 2. CuPy

Install the CuPy package for your CUDA version:

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x

# Or auto-detect
pip install cupy
```

CuPy is **not** listed in `requirements.txt` because it is optional.

---

## Enabling the GPU Backend

Set **one** of the following environment variables before starting the
backend:

```bash
# Option A
export ORBITGUARD_GPU=1

# Option B
export ORBITGUARD_BACKEND=gpu
```

On Windows (PowerShell):

```powershell
$env:ORBITGUARD_GPU = "1"
```

When the backend starts you will see a one-time log line:

```
INFO: OrbitGuard compute backend: GPU (CuPy + CUDA)
```

If CuPy or CUDA is unavailable the backend falls back to NumPy
automatically and logs a warning:

```
WARNING: GPU requested but CuPy/CUDA not available — falling back to CPU (NumPy)
```

---

## Validation

Run the comparison script to verify CPU and GPU produce equivalent
results:

```bash
# CPU baseline
python scripts/compare_cpu_gpu.py

# With GPU
ORBITGUARD_GPU=1 python scripts/compare_cpu_gpu.py
```

The script prints timing and result deltas for Monte Carlo Pc, batch
distance, and TCA norms.

---

## Numerical Differences

| Source | Expected Delta | Reason |
|--------|---------------|--------|
| Monte Carlo Pc | Statistical (≤ 1/√N) | Different RNG sequence on GPU (Cholesky-transform sampling vs NumPy's `multivariate_normal`). Same algorithm, same distribution. |
| Distance / norm | ≤ 1e-14 | IEEE 754 double-precision; operation order may differ. |
| Foster / Chan Pc | 0 | Always runs on CPU. |

For deterministic seeds, CPU and GPU Monte Carlo produce **statistically
equivalent** but not bit-identical results.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'cupy'` | Install CuPy: `pip install cupy-cuda12x` |
| `cupy.cuda.runtime.CUDARuntimeError` | Check `nvidia-smi` — driver/toolkit mismatch |
| GPU enabled but no speedup | Array too small (< 1000 elements); GPU overhead dominates |
| `OutOfMemoryError` | Reduce `n_samples` or use CPU backend |
| Backend falls back to CPU silently | Set log level to INFO to see the one-time message |

---

## Architecture

```
ORBITGUARD_GPU=1
       │
       ▼
 compute/backend.py
   get_xp() ──► cupy (GPU) or numpy (CPU)
   seed_rng()   ► GPU RNG via Cholesky transform
   asnumpy()    ► transfer GPU → CPU
   to_device()  ► transfer CPU → GPU
       │
       ▼
 pc_calculator   ► Monte Carlo on GPU
 screening_engine ► batch norms on GPU
 tca_finder       ► vectorized norms on GPU
```

All public-facing outputs remain Python `float` or `numpy.ndarray` —
CuPy arrays are never exposed outside the compute modules.
