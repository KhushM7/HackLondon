# GPU Acceleration Inventory — OrbitGuard

> Generated during Phase 1 analysis. This document catalogues every numerical
> function in the backend, assesses GPU suitability, and provides recommendations.

---

## 1. GPU-Recommended Functions

| File | Function | What It Computes | Why GPU-Suitable | Current Bottleneck | Suggested GPU Approach | Complexity | Risk | Test Coverage |
|------|----------|-----------------|-----------------|-------------------|----------------------|------------|------|---------------|
| `pc_calculator.py` | `compute_pc_monte_carlo` | Monte Carlo collision probability via bivariate normal sampling + distance threshold | Large batch of independent samples (10k–100k+); embarrassingly parallel sampling, norm computation, and hit counting | `rng.multivariate_normal` + `np.linalg.norm` on N×2 array | **CuPy vectorization**: `cupy.random` for sampling, `cupy.linalg.norm` for distances, `cupy.sum` for hit count | Low | Low — drop-in replacement | ✅ Tested: `test_monte_carlo_agreement_with_foster` |
| `screening_engine.py` | `_closest_approach` (distance computation) | Euclidean distances between two position arrays over a time window | Vectorized `np.linalg.norm(d_pos - i_pos, axis=1)` on arrays up to ~10k×3 (7 days × 60s steps) | Array subtraction + norm on potentially large arrays | **CuPy vectorization**: transfer arrays to GPU, compute distances, transfer scalar result back | Low | Low — isolated vectorized block | ✅ Indirectly via integration tests |
| `tca_finder.py` | `find_tca_from_states` (vectorized block) | Relative distances and dot products from pre-computed state arrays | `np.linalg.norm(r_rel, axis=1)` and `np.sum(r_rel * v_rel, axis=1)` on N×3 arrays | Vectorized NumPy operations on arrays up to ~10k rows | **CuPy vectorization**: same operations using `xp` abstraction | Low | Low — drop-in replacement | ✅ `test_known_geometry`, `test_identical_objects_rejected` |

### Estimated Speedup

| Function | Array Size (typical) | Expected GPU Speedup | Notes |
|----------|---------------------|---------------------|-------|
| `compute_pc_monte_carlo` | 10k–1M samples × 2D | **5–50×** (large N) | GPU RNG + vectorized norm; largest win at N≥100k |
| `_closest_approach` distances | ~7k–10k × 3 | **2–5×** | Moderate arrays; GPU transfer overhead may limit gains for small arrays |
| `find_tca_from_states` | ~7k–10k × 3 | **2–5×** | Same profile as above |

---

## 2. Partially Suitable (Batch Scenarios Only)

| File | Function | What It Computes | Why Partially Suitable | Current Profile | Recommendation |
|------|----------|-----------------|----------------------|-----------------|----------------|
| `covariance_handler.py` | `project_covariance_to_bplane` | 2×3 @ 3×3 @ 3×2 matrix projection | Only 2×2/3×3 matrices; GPU overhead > computation unless batching hundreds of conjunctions | One-off per conjunction | **Keep on CPU** unless batch conjunction processing is added |
| `covariance_handler.py` | `ensure_positive_definite` | Eigendecomposition + Higham correction on small matrices | 2×2 or 6×6 matrices — too small for GPU benefit | One-off per conjunction | **Keep on CPU** |
| `bplane_analysis.py` | `compute_bplane` | Cross products, dot products, norms on 3-vectors | Individual 3-vector operations; no batch dimension | One-off per conjunction | **Keep on CPU** |
| `maneuver_planner.py` | `compute_trade_study` | 6 directions × N delta-v magnitudes | Each iteration involves full ODE propagation; cannot easily batch | Sequential ODE solves dominate | **Keep on CPU** |

---

## 3. Not Recommended for GPU

| File | Function | What It Computes | Why NOT GPU-Suitable |
|------|----------|-----------------|---------------------|
| `pc_calculator.py` | `compute_pc_foster` | Foster 2D Gaussian Pc via `dblquad` | Adaptive quadrature with recursive subdivision; inherently sequential, branchy control flow |
| `pc_calculator.py` | `compute_pc_chan` | Chan maximum Pc (analytical formula) | Single `eigvalsh` on 2×2 + scalar math; trivially fast on CPU |
| `cowell_propagator.py` | `propagate_cowell` | High-fidelity orbit propagation via DOP853 | Adaptive ODE solver with variable step size; each step depends on the previous; sequential by nature |
| `cowell_propagator.py` | `_j2_j6_acceleration` | J2–J6 zonal harmonic accelerations | Single 3-vector computation per RHS evaluation; called inside ODE stepper |
| `cowell_propagator.py` | `_drag_acceleration` | Atmospheric drag acceleration | Single 3-vector + branching (altitude check); inside ODE stepper |
| `cowell_propagator.py` | `_srp_acceleration` | Solar radiation pressure | Trivial scalar computation |
| `tca_finder.py` | `find_tca_sgp4` (coarse loop) | Coarse scan over time window calling SGP4 | SGP4 is a C-extension called sequentially; each call is independent but tiny; overhead of GPU launch > computation |
| `tca_finder.py` | Brent root-finding | Refine TCA to sub-second accuracy | `brentq` is sequential iterative root-finding |
| `maneuver_planner.py` | `optimize_maneuver` | Binary search for minimum delta-v | Sequential; each iteration calls full simulate_maneuver |
| `maneuver_planner.py` | `simulate_maneuver` | Single maneuver simulation | Dominated by Cowell propagation (ODE) |
| `maneuver_planner.py` | `apply_impulsive_burn_sgp4` | SGP4 state extraction + Cowell propagation | SGP4 + ODE solver |
| `propagate_engine.py` | `PropagateEngine.propagate` | SGP4 propagation loop | Sequential SGP4 C calls in a while loop |
| `congestion_analyser.py` | `CongestionAnalyser.compute` | Altitude binning + counting | DB-bound; pure Python dict operations |
| `orbital_constants.py` | `eci_to_rtn` / `rtn_to_eci` | 3×3 rotation matrix from position/velocity | Single 3-vector cross products; trivially fast |
| `tle_validator.py` | `validate_tle` | TLE format + SGP4 parse validation | String parsing + single SGP4 call |
| `ingest_service.py` | TLE ingestion | HTTP download + DB upsert | I/O bound, not compute |

---

## 4. Dependency Analysis

### NumPy Calls That Will Use `xp` Abstraction

| Call Pattern | Files Using It | GPU Migration |
|-------------|---------------|---------------|
| `np.linalg.norm(..., axis=1)` | `screening_engine.py`, `tca_finder.py`, `pc_calculator.py` | ✅ `xp.linalg.norm` drop-in |
| `np.sum(a * b, axis=1)` | `tca_finder.py` | ✅ `xp.sum` drop-in |
| `rng.multivariate_normal(...)` | `pc_calculator.py` | ✅ `cupy.random` equivalent |
| `np.argmin(...)` | `screening_engine.py`, `tca_finder.py` | ✅ `xp.argmin` drop-in |
| `np.array(...)` | All modules | ⚠️ Only convert where arrays flow into GPU-accelerated paths |

### SciPy Calls That Stay on CPU

| Call | File | Reason |
|------|------|--------|
| `scipy.integrate.dblquad` | `pc_calculator.py` | Adaptive quadrature; no CuPy equivalent |
| `scipy.optimize.brentq` | `tca_finder.py` | Sequential root-finding |
| `scipy.optimize.minimize_scalar` | `maneuver_planner.py` | Sequential optimization |
| `scipy.stats.norm` | `pc_calculator.py` | Only used for import, not in hot path |

---

## 5. Architecture: GPU Backend Abstraction

```
backend/app/services/compute_backend.py
├── get_xp()           → returns numpy or cupy
├── is_gpu_enabled()   → bool
├── asnumpy(array)     → always returns numpy array
├── get_rng(seed)      → returns xp-compatible RNG
└── Configuration via:
    ├── ORBITGUARD_BACKEND=cpu|gpu  (env var)
    └── Automatic fallback if CuPy unavailable
```

### Design Principles
- **`xp` pattern**: Functions that benefit from GPU use `xp = get_xp()` and write array-library-agnostic code
- **CPU fallback**: If CuPy not installed or no CUDA device, `get_xp()` returns `numpy`
- **Output contract**: Public API always returns NumPy arrays / Python scalars
- **One-time log**: Backend selection logged once at import time

---

## 6. Implementation Priority Order

### Priority 1: Monte Carlo Pc (`compute_pc_monte_carlo`)
- **Impact**: Highest — called per conjunction with 5k–50k samples
- **Approach**: Replace `rng.multivariate_normal` + `np.linalg.norm` with CuPy equivalents
- **Risk**: Low — self-contained function, well-tested
- **GPU operations**: Cholesky of 2×2 cov, batch random normal sampling, batch distance computation, reduction (sum of hits)

### Priority 2: Screening Distance Computation
- **Impact**: Medium — vectorized distance over ~10k timestep array
- **Approach**: Use `xp.linalg.norm` in `_closest_approach`
- **Risk**: Low — isolated computation block

### Priority 3: TCA State Analysis
- **Impact**: Medium — vectorized distance + dot product
- **Approach**: Use `xp` in `find_tca_from_states`
- **Risk**: Low — same profile as Priority 2

---

## 7. Numerical Tolerance Expectations

| Computation | CPU vs GPU Expected Difference | Acceptable Tolerance |
|------------|-------------------------------|---------------------|
| Monte Carlo Pc | Statistical variation from different RNG sequences | Within Wilson score 95% CI |
| Distance norms | IEEE 754 float64 rounding differences | < 1e-12 relative error |
| Dot products | Reduction order may differ | < 1e-12 relative error |
| Argmin/argmax | Identical (integer results) | Exact match |

---

## 8. Files to Modify

| File | Change Type | Description |
|------|------------|-------------|
| `backend/app/services/compute_backend.py` | **NEW** | Backend abstraction module |
| `backend/app/services/pc_calculator.py` | **MODIFY** | GPU path for Monte Carlo |
| `backend/app/services/screening_engine.py` | **MODIFY** | GPU path for distance computation |
| `backend/app/services/tca_finder.py` | **MODIFY** | GPU path for vectorized state analysis |
| `backend/requirements.txt` | **MODIFY** | Add cupy as optional dependency |
| `backend/tests/test_gpu_backend.py` | **NEW** | Backend + parity tests |
| `scripts/compare_cpu_gpu.py` | **NEW** | Validation/benchmark script |
| `docs/GPU.md` | **NEW** | GPU setup and usage documentation |
