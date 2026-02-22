# GPU Acceleration Inventory — OrbitGuard

> Generated 2026-02-22. This document catalogs every numerically significant function in the OrbitGuard backend, evaluates GPU suitability, and recommends an acceleration strategy.

---

## 1. Functions Recommended for GPU Acceleration

| # | File | Function | What It Computes | Why GPU-Suitable | Current Bottleneck | Suggested GPU Approach | Risk | Test Coverage |
|---|------|----------|------------------|------------------|--------------------|------------------------|------|---------------|
| 1 | `pc_calculator.py` | `compute_pc_monte_carlo` | Monte Carlo collision probability via N random samples from bivariate normal; counts hits inside HBR disk | **High parallelism**: N independent samples (default 10 000), each needing: (a) sample from multivariate normal, (b) compute norm, (c) compare to threshold. All are embarrassingly parallel. | `rng.multivariate_normal` + `np.linalg.norm` over N samples. For large N (50 000+), this dominates. | **CuPy vectorization**: Use `cupy.random` for batched sampling, `cupy.linalg.norm` for distances, `cupy.sum` for hit count. Drop-in replacement via `xp` abstraction. | Low — algorithm is identical; only floating-point RNG ordering differs. | ✅ `test_orbital_mechanics.py::TestPcCalculator::test_monte_carlo_agreement_with_foster` |
| 2 | `screening_engine.py` | `_closest_approach` (distance computation block) | Batch `np.linalg.norm(d_pos - i_pos, axis=1)` over all propagation time steps (~7000+ samples at 60 s / 7 days) | **Vectorized array op** on large arrays: subtract two (N,3) arrays, row-wise norm. Purely data-parallel. | Lines 103-124: array construction + `np.linalg.norm` | **CuPy vectorization**: Convert position arrays to GPU, compute norms on GPU, `argmin` on GPU. | Low — pure array math, no branching. | ✅ Implicitly via `test_screening_logic.py` and integration tests |
| 3 | `tca_finder.py` | `find_tca_from_states` (vectorized portion) | Relative position/velocity norms and dot products over N time steps | Lines 243-247: `np.linalg.norm(r_rel, axis=1)` and `np.sum(r_rel * v_rel, axis=1)` — both fully vectorizable on (N,3) arrays. | Norm + dot product on ~7000 rows | **CuPy vectorization**: Same `xp` pattern. | Low | ✅ `test_orbital_mechanics.py::TestTCAFinder` |
| 4 | `tca_finder.py` | `find_tca_sgp4` (coarse scan loop) | Computes N distances and N dot products via repeated SGP4 calls in a Python loop (~10 000 iterations for 168 h at 60 s) | The *distance/dot-product math* is GPU-suitable, but the SGP4 calls are serial C-library calls and cannot be GPU-accelerated. **Only** the post-collection norm/argmin is worth porting. | Python loop calling SGP4 + per-step distance. SGP4 call dominates. | **Partial CuPy**: After collecting arrays, compute `np.linalg.norm` and `np.argmin` on GPU. Loop itself stays CPU. | Low | ✅ Implicitly via screening tests |
| 5 | `covariance_handler.py` | `ensure_positive_definite` | Eigendecomposition + clip + reconstruct for covariance repair | Only beneficial if called in a **batch** (many conjunctions processed together). Single 6×6 or 2×2 matrix is too small for GPU. | `np.linalg.eigh` on small matrix | **CuPy vectorization (batched only)**: If processing 100+ conjunctions, batch all matrices into (B,6,6) and use `cupy.linalg.eigh`. Otherwise keep CPU. | Medium — need to restructure calling code for batching | ✅ `TestCovarianceHandler::test_higham_correction` |
| 6 | `covariance_handler.py` | `project_covariance_to_bplane` | `R @ C @ R^T` matrix product (2×3 @ 3×3 @ 3×2) | Trivial for single call. Beneficial only in **batch** over many conjunctions. | Matrix multiply on 3×3 | **Batched CuPy `@` operator** if processing many conjunctions simultaneously. | Medium | ✅ `TestCovarianceHandler::test_bplane_projection` |
| 7 | `screening_engine.py` | `find_conjunctions` (outer loop) | Iterates over all candidate objects, propagates each, finds closest approach | The **inner calls** (propagate + distance) could be parallelized across candidates on GPU if SGP4 supported it. Currently sequential. | Python for-loop over candidates (line 54-67) | **Future**: Batch all candidate propagations if a GPU-native propagator is used. For now, only the per-candidate distance math benefits. | High — requires architectural change | ✅ Integration tests |

---

## 2. Functions NOT Recommended for GPU Acceleration

| # | File | Function | What It Computes | Why NOT GPU-Suitable | Recommendation |
|---|------|----------|------------------|----------------------|----------------|
| 1 | `cowell_propagator.py` | `propagate_cowell` | Adaptive ODE integration (DOP853) with perturbations | **Adaptive step-size ODE solver** (SciPy `solve_ivp`): inherently sequential — each step depends on the previous. Branching in perturbation selection (drag on/off, SRP on/off). State vector is only 6 elements. | **Keep on CPU**. No GPU benefit for single-trajectory adaptive integration. |
| 2 | `cowell_propagator.py` | `_j2_j6_acceleration`, `_drag_acceleration`, `_srp_acceleration` | Force model evaluations per integration step | Called inside ODE RHS — single (3,) vector operations. GPU kernel launch overhead >> computation time. | **Keep on CPU**. |
| 3 | `pc_calculator.py` | `compute_pc_foster` | 2D Gaussian integration via `scipy.integrate.dblquad` | **Adaptive quadrature**: recursive subdivision with error control. Cannot be parallelized on GPU. Integrand is evaluated one point at a time by the SciPy algorithm. | **Keep on CPU**. Could consider batching across many conjunctions in future. |
| 4 | `pc_calculator.py` | `compute_pc_chan` | Closed-form Chan upper-bound Pc | Single scalar computation: `eigvalsh` on 2×2 + one `exp`. Sub-microsecond on CPU. | **Keep on CPU**. |
| 5 | `tca_finder.py` | `find_tca_sgp4` (Brent root-finding) | Brent's method refinement of TCA | **Sequential root-finding**: each function evaluation depends on the result of the bracket update. Cannot be parallelized. | **Keep on CPU**. |
| 6 | `tca_finder.py` | `_sgp4_state_at`, `_relative_dot_product`, `_distance_at` | Single SGP4 propagation + vector ops | SGP4 is a compiled C library; GPU would require a CUDA reimplementation. Single-point ops are too small. | **Keep on CPU**. |
| 7 | `propagate_engine.py` | `PropagateEngine.propagate` | Sequential SGP4 propagation loop | Python while-loop calling C-library SGP4. Each call is fast but serial. GPU would require porting SGP4 to CUDA. | **Keep on CPU**. |
| 8 | `bplane_analysis.py` | `compute_bplane` | B-plane frame construction: cross products, norms, dot products on (3,) vectors | Tiny vectors (3 elements). GPU kernel launch latency >> computation. | **Keep on CPU**. |
| 9 | `orbital_constants.py` | `eci_to_rtn`, `rtn_to_eci` | 3×3 rotation matrix from position/velocity | Single 3-vector operations. | **Keep on CPU**. |
| 10 | `maneuver_planner.py` | `simulate_maneuver`, `optimize_maneuver`, `compute_trade_study` | Maneuver simulation pipeline with Pc re-evaluation | Orchestration code calling propagator + Pc. The inner Pc calls benefit indirectly from GPU Monte Carlo, but the outer loop (binary search, trade study iteration) is inherently serial. | **Keep on CPU** (benefits from GPU-accelerated inner calls). |
| 11 | `covariance_handler.py` | `estimate_covariance_from_tle`, `default_covariance` | Construct diagonal covariance matrices | Scalar arithmetic + `np.diag`. Trivial. | **Keep on CPU**. |
| 12 | `ingest_service.py` | All functions | TLE parsing, HTTP fetch, DB upsert | I/O-bound, no numerical computation. | **Keep on CPU**. |
| 13 | `congestion_analyser.py` | `CongestionAnalyser.compute` | Altitude binning and counting | DB queries + Python dict counting. Not numerical. | **Keep on CPU**. |
| 14 | `tle_validator.py` | All functions | TLE format validation and checksum | String parsing. Not numerical. | **Keep on CPU**. |
| 15 | `avoidance_simulator.py` | All functions | Orchestration wrapper over maneuver_planner | DB + delegation. Not numerical. | **Keep on CPU**. |

---

## 3. Priority Ranking

### Priority 1 — Monte Carlo Pc (Highest Impact)
- **Function**: `compute_pc_monte_carlo`
- **Why**: 10 000–50 000 independent samples, each doing multivariate normal draw + norm + comparison. Embarrassingly parallel. This is the classic GPU workload.
- **Expected speedup**: **10–50×** for N ≥ 10 000 samples (GPU RNG + vectorized norm).
- **Complexity**: Low — drop-in `xp` replacement.
- **Input sizes**: `n_samples` = 5 000–50 000 (configurable). 2D samples.

### Priority 2 — Batch Distance Computation in Screening
- **Function**: `screening_engine._closest_approach` distance block
- **Why**: `np.linalg.norm(d_pos - i_pos, axis=1)` on arrays of shape (N, 3) where N ≈ 7 000+ (7 days × 60 s resolution). Pure SIMD.
- **Expected speedup**: **2–5×** (array is medium-sized; GPU shines more at N > 10 000).
- **Complexity**: Low.

### Priority 3 — TCA State-Based Finder Vectorized Portion
- **Function**: `find_tca_from_states` norm/dot-product block
- **Why**: Same pattern as Priority 2. Arrays of (N, 3).
- **Expected speedup**: **2–5×**.
- **Complexity**: Low.

### Priority 4 (Future) — Batched Covariance Operations
- **Functions**: `ensure_positive_definite`, `project_covariance_to_bplane`
- **Why**: Only beneficial when processing many conjunctions simultaneously (batch eigendecomposition on GPU).
- **Expected speedup**: **5–20×** if batched across 100+ conjunctions.
- **Complexity**: Medium — requires restructuring caller to batch.
- **Recommendation**: Defer to Phase 2.

---

## 4. Dependencies

| Dependency | Used By | GPU Equivalent |
|-----------|---------|----------------|
| `numpy` | All numerical code | `cupy` (drop-in) |
| `numpy.random.default_rng` | Monte Carlo sampling | `cupy.random` (different API, needs wrapper) |
| `numpy.linalg.norm` | Distance computations | `cupy.linalg.norm` |
| `numpy.linalg.eigvalsh` | Covariance PD check | `cupy.linalg.eigvalsh` |
| `numpy.linalg.cholesky` | Covariance PD check | `cupy.linalg.cholesky` |
| `numpy.linalg.det` | Foster Pc | Keep on CPU (single 2×2) |
| `numpy.linalg.inv` | Foster Pc | Keep on CPU (single 2×2) |
| `scipy.integrate.dblquad` | Foster Pc | **No GPU equivalent** — keep on CPU |
| `scipy.optimize.brentq` | TCA root-finding | **No GPU equivalent** — keep on CPU |
| `scipy.integrate.solve_ivp` | Cowell propagator | **No GPU equivalent** — keep on CPU |
| `scipy.optimize.minimize_scalar` | Maneuver optimizer | **No GPU equivalent** — keep on CPU |
| `scipy.stats.norm` | Wilson CI (Monte Carlo) | Keep on CPU (single scalar) |
| `sgp4` | All propagation | C library, **no GPU equivalent** |

---

## 5. Architecture Summary

```
┌──────────────────────────────────────────────┐
│           Environment / Config               │
│  ORBITGUARD_GPU=1  or  ORBITGUARD_BACKEND=gpu │
└─────────────────┬────────────────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │   compute/backend.py      │
    │   get_xp() → numpy|cupy  │
    │   is_gpu_enabled() → bool │
    │   asnumpy(arr) → np.array │
    │   to_device(arr) → xp.arr │
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────────────────────────────────┐
    │                  GPU-Accelerated Modules               │
    │                                                        │
    │  pc_calculator.py     → Monte Carlo sampling + norms   │
    │  screening_engine.py  → Batch distance computation     │
    │  tca_finder.py        → Batch norm + dot products      │
    └────────────────────────────────────────────────────────┘
                  │
    ┌─────────────▼─────────────────────────────────────────┐
    │                  CPU-Only Modules                       │
    │                                                        │
    │  cowell_propagator.py → Adaptive ODE (DOP853)          │
    │  pc_calculator.py     → Foster (dblquad), Chan          │
    │  tca_finder.py        → Brent root-finding              │
    │  bplane_analysis.py   → Small vector ops                │
    │  maneuver_planner.py  → Orchestration                   │
    │  propagate_engine.py  → SGP4 (C library)                │
    └────────────────────────────────────────────────────────┘
```

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| CuPy RNG produces different sequences than NumPy for same seed | Accept statistical equivalence; do not require bit-exact match. Test within tolerance bands. |
| GPU kernel launch overhead exceeds benefit for small arrays | Only use GPU when array size exceeds threshold (e.g., N > 1000). Fall back to CPU for small inputs. |
| CuPy not installed / no CUDA device | Graceful fallback: `get_xp()` returns `numpy` and logs one-time warning. |
| Numerical precision differences (float32 vs float64) | Always use float64 on GPU (CuPy default). No precision downgrade. |
| GPU memory exhaustion for very large sample counts | CuPy raises `OutOfMemoryError`; catch and fall back to CPU. |

---

## 7. Estimated Effort

| Task | Complexity | Files Changed |
|------|-----------|---------------|
| Backend abstraction module | Low | 1 new file |
| Monte Carlo GPU port | Low | 1 file modified |
| Screening distance GPU port | Low | 1 file modified |
| TCA finder GPU port | Low | 1 file modified |
| Tests | Low-Medium | 1-2 new test files |
| Validation script | Low | 1 new file |
| Documentation | Low | 2 new files |
