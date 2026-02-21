# OrbitGuard (Hackathon MVP)

OrbitGuard is a public-data powered satellite conjunction screening and visualisation platform.

## What is implemented

- FastAPI backend with TLE ingestion, propagation, conjunction screening, avoidance simulation, and congestion analysis
- REST API endpoints from PRD: `/catalog`, `/assets/{norad_id}/conjunctions`, `/conjunctions/{event_id}`, `/conjunctions/{event_id}/avoidance-sim`, `/congestion`
- Extra endpoint for manual refresh: `POST /ingest/refresh`
- Internal ingestion scheduler (every 6 hours by default)
- Next.js frontend with:
  - Dashboard
  - Asset View
  - Conjunction Event View with 3D replay and pre/post manoeuvre toggle
  - Congestion chart
- Transparency messaging in backend health response and frontend UI

## Repo layout

- `backend/`: FastAPI app and core services
- `frontend/`: Next.js app
- `progress.md`: build checklist and progress tracking
- `prd.md`: source requirements

## Backend setup

1. Create a virtual environment and install dependencies:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Copy env file and adjust values if needed:
```bash
cp .env.example .env
```
3. Run API:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend setup

1. Install dependencies:
```bash
cd frontend
npm install
```
2. Copy env file:
```bash
cp .env.example .env.local
```
`API_BASE` is used server-side (Next SSR) and `NEXT_PUBLIC_API_BASE` is used client-side. Set both to the same backend URL (for example `http://127.0.0.1:8000`) to avoid fetch host mismatch.
3. Start frontend:
```bash
npm run dev
```

## Notes and constraints

- Data source is public TLE only.
- This is a screening-grade system.
- No formal collision probability (Pc) is computed.
- Avoidance simulation is simplified and non-authoritative.
