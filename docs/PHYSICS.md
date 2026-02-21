# OrbitGuard Physics Documentation

## Mathematical Foundations

This document describes the orbital mechanics, collision probability theory, and maneuver planning algorithms used in OrbitGuard's high-fidelity conjunction analysis system.

---

## 1. Orbit Propagation

### 1.1 SGP4/SDP4 (Primary Method)

Two-Line Element (TLE) sets are propagated using the SGP4/SDP4 analytical theory via the `sgp4` library. SGP4 models:

- Secular and periodic effects of J2 (Earth oblateness)
- Atmospheric drag via the B* parameter
- Deep-space perturbations (SDP4) for periods > 225 minutes

Output frame: TEME (True Equator Mean Equinox).

**Reference:** Vallado et al. (2006), "Revisiting Spacetrack Report #3"

### 1.2 Cowell Numerical Propagation (High-Fidelity)

For maneuver planning and close-approach refinement, we use Cowell's method — direct numerical integration of the equations of motion:

```
r̈ = -μ/r³ · r + a_perturbations
```

Integrated using SciPy's DOP853 (8th-order Dormand-Prince) with adaptive step control.

#### Perturbation Models

**J2-J6 Zonal Harmonics** (EGM2008 coefficients):

```
a_J2 = (3μJ₂R²ₑ)/(2r⁵) · [x(1-5z²/r²), y(1-5z²/r²), z(3-5z²/r²)]
```

Higher-order terms (J3-J6) follow Vallado (2013) Eq. 8-25 through 8-30.

**Atmospheric Drag** (exponential density model):

```
a_drag = -½ · ρ · Cd · (A/m) · |v_rel|² · v̂_rel
ρ = ρ₀ · exp(-(h - h₀) / H)
```

Where ρ₀ = 3.614×10⁻¹³ kg/m³ at h₀ = 700 km, scale height H = 88.667 km.

**Solar Radiation Pressure:**

```
a_SRP = -Cr · (A/m) · (S/c) · ŝ
```

Where S = 1361 W/m², c = speed of light, ŝ = sun direction unit vector.

---

## 2. Time of Closest Approach (TCA)

### Algorithm

TCA is found by solving for the time where the range rate is zero:

```
d/dt |r_rel|² = 2 · r_rel · v_rel = 0
```

**Procedure:**

1. **Coarse scan:** Evaluate r_rel · v_rel at each time step (60s default).
2. **Sign change detection:** Find intervals where r_rel · v_rel transitions from negative (approaching) to positive (receding).
3. **Root refinement:** Apply Brent's method (`scipy.optimize.brentq`) to find the exact zero crossing with sub-second accuracy (xtol = 0.01 s).
4. **Multi-TCA handling:** If multiple close approaches exist within the window, the one with minimum miss distance is selected.

**Reference:** Vallado (2013), Chapter 10.

---

## 3. B-Plane Analysis

The B-plane (impact parameter plane) is the standard reference frame for conjunction geometry, used in Conjunction Data Messages (CDMs) per CCSDS 508.0-B-1.

### Frame Construction

At TCA, the B-plane coordinate system is defined as:

```
T̂ = v_rel / |v_rel|         (relative velocity unit vector, normal to B-plane)
N̂ = T̂ × K̂ / |T̂ × K̂|     (K̂ = Earth's pole [0,0,1])
R̂ = T̂ × N̂                  (completes right-handed system)
```

### B-Vector

The miss vector is projected onto the B-plane:

```
B = r_miss - (r_miss · T̂) · T̂
B·T = B · N̂     (in-plane component)
B·R = B · R̂     (out-of-plane component)
|B| = √(B·T² + B·R²)
θ = atan2(B·R, B·T)
```

**Reference:** Vallado (2013), Section 10.4; CCSDS 508.0-B-1.

---

## 4. Probability of Collision (Pc)

### 4.1 Foster Method (2D Gaussian — Operational Standard)

The primary Pc method, used by NASA CARA and most SSA providers.

**Formulation:**

```
Pc = ∬_{disk(HBR)} f(x; μ, Σ) dA
```

Where:
- μ = [B·T, B·R] is the miss vector in the B-plane
- Σ = C_primary + C_secondary is the combined covariance projected to the B-plane (2×2)
- HBR = r₁ + r₂ is the Hard Body Radius (sum of physical radii)
- f is the bivariate normal PDF

The integral is computed numerically using `scipy.integrate.dblquad`.

**Special case (zero miss, isotropic covariance):**
```
Pc = 1 - exp(-HBR² / (2σ²))
```

**Reference:** Alfano (2005); Foster & Estes (1992).

### 4.2 Chan Method (Maximum Pc)

Provides an upper bound when the full B-plane geometry is uncertain:

```
Pc_max = HBR² / (2·σ_max·σ_min) · exp(-d² / (2·σ_max²))
```

Where σ_max, σ_min are the square roots of the eigenvalues of the B-plane covariance and d is the scalar miss distance.

**Reference:** Chan (1997).

### 4.3 Monte Carlo Method

For validation and high-uncertainty scenarios:

1. Draw N samples from the combined bivariate normal distribution.
2. Count the fraction within the HBR disk.
3. Report Pc with Wilson score 95% confidence interval.

Default: N = 10,000 samples.

---

## 5. Covariance Handling

### 5.1 Sources

1. **CDM-provided:** Full 6×6 or 3×3 position covariance.
2. **TLE-estimated (Vallado-Cefola model):** Covariance estimated from TLE epoch age and B* drag term:
   - σ_pos = 1 km × (1 + 0.5·age + 0.1·age²) × (1 + |B*|×10⁴)
   - Along-track uncertainty is 3× cross-track
   - Radial uncertainty is 0.5× cross-track
3. **Default:** Isotropic 1 km position uncertainty.

### 5.2 Positive Definite Validation

All covariance matrices are checked for positive definiteness via Cholesky decomposition. If not PD, the Higham (1988) nearest-PD correction is applied:

1. Compute eigendecomposition: C = V·Λ·Vᵀ
2. Clip negative eigenvalues: λᵢ = max(λᵢ, ε) where ε = 10⁻¹⁰
3. Reconstruct: C' = V·Λ'·Vᵀ

### 5.3 B-Plane Projection

Combined covariance is projected to the B-plane using the rotation matrix R = [N̂; R̂]:

```
C_bplane = R · (C_primary + C_secondary) · Rᵀ
```

---

## 6. Maneuver Planning

### 6.1 RTN Frame

All burns are expressed in the Radial-Tangential-Normal frame:

- **R (Radial):** Toward Earth center
- **T (Tangential):** Along velocity (prograde for circular orbits)
- **N (Normal):** Orbit normal (cross-track)

Conversion: RTN → ECI via rotation matrix from the satellite's position and velocity vectors.

### 6.2 Impulsive Burn Model

A delta-v burn is applied as an instantaneous velocity change at a specified epoch:

```
v_new = v_old + R_RTN→ECI · Δv_RTN
```

The post-burn orbit is then propagated using Cowell's method to find the new TCA and Pc.

### 6.3 Minimum Delta-V Optimization

Binary search over delta-v magnitude to find the smallest burn that achieves Pc < target:

1. Check if maximum delta-v achieves the target.
2. Binary search (20 iterations → ~10⁻⁶ m/s precision).
3. Return the optimal burn with full post-maneuver analysis.

### 6.4 Fuel Cost (Tsiolkovsky Equation)

```
Δm = m₀ · (1 - exp(-Δv / (Isp · g₀)))
```

Where:
- m₀ = satellite mass [kg]
- Isp = specific impulse [s] (default: 300 s for bipropellant)
- g₀ = 9.80665 m/s²

### 6.5 Deorbit Safety Check

After applying a burn, the resulting perigee altitude is checked:

```
a = -μ / (2ε)     where ε = ½v² - μ/r
e = √(1 + 2εh²/μ²)
r_perigee = a(1-e) - R_earth
```

If r_perigee < 100 km, the maneuver is rejected.

---

## 7. Risk Classification

Per NASA CARA / ESA operational standards:

| Level  | Pc Range         | Action             |
|--------|------------------|---------------------|
| GREEN  | Pc < 10⁻⁵       | Monitor             |
| YELLOW | 10⁻⁵ ≤ Pc < 10⁻⁴ | Elevated monitoring |
| RED    | Pc ≥ 10⁻⁴       | Maneuver required   |

---

## 8. Units Convention

| Quantity    | Internal Unit | API Unit |
|-------------|---------------|----------|
| Distance    | km            | km       |
| Velocity    | km/s          | km/s     |
| Time        | seconds       | ISO 8601 |
| Angles      | radians       | degrees (display only) |
| Pc          | dimensionless | dimensionless (0-1) |
| Mass        | kg            | kg       |
| Delta-v     | m/s           | m/s      |

---

## References

1. Alfano, S. (2005). "A Numerical Implementation of Spherical Object Collision Probability." *Journal of the Astronautical Sciences*, 53(1), 103-109.
2. Chan, F. K. (1997). *Spacecraft Collision Probability.* The Aerospace Press.
3. Foster, J. L. & Estes, H. S. (1992). "A Parametric Analysis of Orbital Debris Collision Probability and Maneuver Rate for Space Vehicles." NASA JSC-25898.
4. Higham, N. J. (1988). "Computing a nearest symmetric positive semidefinite matrix." *Linear Algebra and its Applications*, 103, 103-118.
5. Patera, R. P. (2005). "Satellite Collision Probability for Nonlinear Relative Motion." *Journal of Guidance, Control, and Dynamics*, 28(4), 642-652.
6. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*, 4th Edition. Microcosm Press.
7. CCSDS 508.0-B-1. "Conjunction Data Message." Consultative Committee for Space Data Systems.
