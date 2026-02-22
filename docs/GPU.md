# GPU Acceleration Guide — OrbitGuard

## Overview

OrbitGuard supports optional GPU acceleration for computationally intensive
numerical operations. When enabled, the following operations run on the GPU
via [CuPy](https://cupy.dev/):

- **Monte Carlo collision probability** — sampling and distance computation
- **Bulk distance screening** — vectorized norm computation across time windows
- **TCA state analysis** — vectorized distance and dot product computation

All other operations (ODE propagation, root-finding, quadrature integration)
remain on the CPU where they perform best.

## Installation

### 1. Install CUDA Toolkit

CuPy requires an NVIDIA GPU with CUDA support. Check your CUDA version:

```bash
nvidia-smi
```

### 2. Install CuPy

Choose the package matching your CUDA version:

```bash
# CUDA 12.x (most common on modern systems)
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x

# Auto-detect (slower install, compiles from source)
pip install cupy
```

### 3. Verify Installation

```python
import cupy
print(cupy.__version__)
print(cupy.cuda.Device(0).compute_capability)
```

## Enabling GPU Backend

Set the environment variable before starting OrbitGuard:

```bash
# Enable GPU
export ORBITGUARD_BACKEND=gpu

# Default: CPU
export ORBITGUARD_BACKEND=cpu
```

The backend is selected once at startup. A log message confirms the active backend:

```
INFO: OrbitGuard compute backend: GPU (CuPy 13.x.x)
```

### Graceful Fallback

If `ORBITGUARD_BACKEND=gpu` is set but CuPy is not installed or no CUDA device
is available, OrbitGuard automatically falls back to CPU with a warning:

```
WARNING: ORBITGUARD_BACKEND=gpu but CuPy is not installed; falling back to CPU.
```

## What Gets Accelerated

| Component | GPU Operation | Expected Speedup |
|-----------|--------------|-----------------|
| Monte Carlo Pc | Random sampling + distance threshold | 5–50× at N≥100k samples |
| Screening distance | `norm(d_pos - i_pos)` over time window | 2–5× for arrays > 5k |
| TCA state analysis | Vectorized distance + dot products | 2–5× for arrays > 5k |

### What Stays on CPU

- Foster 2D Gaussian Pc (adaptive quadrature — `dblquad`)
- Chan maximum Pc (trivial analytical formula)
- Cowell orbit propagation (adaptive ODE solver — DOP853)
- TCA root-finding (Brent's method)
- Maneuver optimization (sequential binary search)
- SGP4 propagation (C extension, sequential)

## Numerical Differences

GPU and CPU results may differ slightly due to:

- **Floating-point reduction order**: GPU parallel reductions accumulate in
  a different order than sequential CPU operations, causing ~1e-14 relative
  differences in norms and dot products.
- **RNG sequences**: GPU and CPU random number generators produce different
  sequences even with the same seed. Monte Carlo results will differ
  statistically but should agree within the Wilson score confidence interval.

### Tolerances

| Computation | Expected Difference | Tolerance |
|------------|-------------------|-----------|
| Distance norms | ~1e-14 relative | < 1e-12 |
| Dot products | ~1e-14 relative | < 1e-12 |
| Monte Carlo Pc | Statistical variation | Within 95% CI |

## Validation

Run the comparison script to verify GPU results match CPU:

```bash
python scripts/compare_cpu_gpu.py
```

This prints timing and numerical deltas for Monte Carlo and distance
computations on both backends.

## Troubleshooting

### CuPy import fails

```
ImportError: No module named 'cupy'
```

→ Install CuPy: `pip install cupy-cuda12x`

### No CUDA device

```
cupy.cuda.runtime.CUDARuntimeError: cudaErrorNoDevice
```

→ Check that NVIDIA drivers are installed: `nvidia-smi`
→ Ensure you have a CUDA-capable GPU

### Wrong CUDA version

```
cupy.cuda.compiler.CompileException
```

→ Install the CuPy package matching your CUDA version
→ Check: `nvcc --version` or `nvidia-smi` for CUDA version

### Performance regression on small arrays

GPU acceleration adds transfer overhead. For arrays < 1000 elements,
CPU may be faster. The default Monte Carlo sample count (5000–10000)
is large enough to benefit from GPU.

## Architecture

```
backend/app/services/compute_backend.py
├── get_xp()           → returns numpy or cupy
├── is_gpu_enabled()   → bool
├── asnumpy(array)     → always returns numpy array
├── get_rng(seed)      → returns xp-compatible RNG
└── reset_backend()    → for testing only
```

GPU-accelerated functions use the `xp = get_xp()` pattern to write
array-library-agnostic code that works with both NumPy and CuPy.
