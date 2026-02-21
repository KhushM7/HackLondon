# OrbitGuard

OrbitGuard is a public-data powered satellite conjunction screening, collision probability analysis, and avoidance planning platform.

## Capabilities

- **TLE Ingestion:** Automatic ingestion from CelesTrak (every 6 hours) with manual refresh
- **SGP4 Propagation:** TLE-based orbit propagation over configurable windows
- **Cowell Numerical Propagation:** High-fidelity propagation with J2-J6 harmonics, drag, SRP perturbations
- **TCA Root-Finding:** Sub-second precision closest approach detection using Brent's method
- **B-Plane Analysis:** Full B-plane geometry (BdotT, BdotR) per CCSDS 508.0-B-1
- **Probability of Collision:** Three methods — Foster (2D Gaussian), Chan (maximum Pc), Monte Carlo with confidence intervals
- **Covariance Handling:** TLE-based estimation, positive-definite validation, Higham correction
- **Maneuver Planning:** RTN frame impulsive burns, minimum delta-v optimizer, 6-direction trade study
- **Fuel Cost:** Tsiolkovsky rocket equation with configurable Isp
- **Risk Classification:** NASA CARA operational thresholds (GREEN/YELLOW/RED)
- **3D Visualization:** Three.js-based event replay with pre/post-maneuver toggle
- **Congestion Analysis:** Altitude-band density and conjunction rate metrics

## Repo layout

- `backend/` — FastAPI app with orbital mechanics core
- `frontend/` — Next.js app with 3D visualization
- `docs/PHYSICS.md` — Mathematical foundations and equations
- `docs/VALIDATION.md` — Test results against reference implementations
- `prd.md` — Product requirements

## Backend setup

1. Create a virtual environment and install dependencies:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
2. Verify dependencies:
```bash
python verify_dependencies.py
```
3. Run tests:
```bash
python -m pytest tests/ -v
```
4. Run API:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/catalog` | List tracked satellites |
| POST | `/catalog/satellite` | Add satellite by NORAD ID |
| GET | `/catalog/positions` | Current positions for all tracked objects |
| GET | `/assets/{norad_id}/conjunctions` | Find conjunctions for a defended asset |
| GET | `/conjunctions/{event_id}` | Full conjunction event detail (B-plane, Pc, covariance) |
| POST | `/conjunctions/{event_id}/avoidance-sim` | Simulate an avoidance maneuver |
| POST | `/events/{event_id}/optimize-maneuver` | Find minimum delta-v for target Pc |
| GET | `/events/{event_id}/trade-study` | 6-direction maneuver trade table |
| GET | `/congestion` | Altitude congestion metrics |
| POST | `/ingest/refresh` | Manual TLE refresh |

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

## Units Convention

- Distances: **km**
- Velocities: **km/s**
- Delta-v: **m/s**
- Probability: raw float (0-1)
- Mass: **kg**
- Time: ISO 8601 UTC

## Architecture

```
backend/app/services/
├── orbital_constants.py    # Physical constants, PropagationConfig, RTN frame
├── cowell_propagator.py    # Numerical integration with perturbations
├── tca_finder.py           # Root-finding for closest approach
├── bplane_analysis.py      # B-plane geometry computation
├── covariance_handler.py   # Covariance estimation, validation, projection
├── pc_calculator.py        # Foster, Chan, and Monte Carlo Pc methods
├── maneuver_planner.py     # RTN burns, optimizer, trade study, fuel cost
├── avoidance_simulator.py  # API-facing simulator (wraps maneuver_planner)
├── screening_engine.py     # Full pipeline: propagate → TCA → B-plane → Pc
├── propagate_engine.py     # SGP4 propagation
├── ingest_service.py       # TLE ingestion from CelesTrak
└── congestion_analyser.py  # Altitude band congestion metrics
```
