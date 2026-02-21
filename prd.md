Below is a structured **Product Requirements Document (PRD)** suitable for a hackathon build that still positions the product as future-grade space traffic infrastructure.

---

# Product Requirements Document

## Product Name: OrbitGuard

## Version: Hackathon MVP

## Document Status: Draft for Build

---

# 1. Product Overview

## 1.1 Vision

OrbitGuard is a public-data powered satellite conjunction screening and visualisation platform that demonstrates how space traffic coordination could function in an increasingly congested Low Earth Orbit environment.

The system detects potential close approaches between tracked objects, visualises them in 3D, and simulates simplified avoidance manoeuvres.

The long-term vision is to evolve into infrastructure supporting scalable space traffic coordination.

---

# 2. Problem Statement

Low Earth Orbit is rapidly becoming congested due to mega-constellations and debris accumulation.

Publicly available orbit data exists, but:

* It is not presented in an actionable way.
* There is no accessible public-facing decision support layer.
* Collision risk is poorly visualised for non-specialists.
* There is limited demonstration of automated avoidance reasoning.

OrbitGuard addresses this by converting raw orbit elements into interpretable conjunction alerts and simulated mitigation options.

---

# 3. Objectives

### 3.1 Primary Objective (Hackathon)

Deliver a working web platform that:

* Detects conjunctions using public TLE data.
* Displays upcoming close approaches for selected assets.
* Visualises events in 3D.
* Simulates a basic avoidance manoeuvre.
* Demonstrates orbital congestion metrics.

### 3.2 Secondary Objective

Position the system as a foundation for:

* Decision-support tooling
* Constellation-scale traffic modelling
* Congestion forecasting
* Future coordination infrastructure

---

# 4. Target Users

### 4.1 Hackathon Judges

Primary audience for MVP demo.

### 4.2 Secondary Personas

* Aerospace engineering students
* Policy analysts
* Space-tech founders
* SSA researchers

---

# 5. Scope

## 5.1 In Scope (MVP)

* LEO satellites only
* Public TLE ingestion
* SGP4 propagation
* Close-approach screening
* Conjunction event database
* 3D event replay
* Simplified avoidance simulation
* Altitude congestion index
* REST API backend
* Web frontend dashboard

## 5.2 Out of Scope (MVP)

* Covariance modelling
* Formal Probability of Collision calculation
* Authoritative manoeuvre recommendation
* Real-time ground sensor fusion
* GEO/MEO support
* Authentication and multi-user management

---

# 6. Functional Requirements

## 6.1 Data Ingestion

FR1: System shall ingest TLE data from CelesTrak every 6 hours.
FR2: System shall store latest TLE per NORAD ID.
FR3: System shall support manual refresh trigger.

---

## 6.2 Orbit Propagation

FR4: System shall propagate orbits using SGP4.
FR5: Propagation horizon shall be 7 days.
FR6: Time resolution shall be 60 seconds (configurable).

---

## 6.3 Conjunction Screening

FR7: System shall screen objects within altitude and inclination proximity bands before detailed comparison.
FR8: System shall compute time of closest approach (TCA).
FR9: System shall compute miss distance in km.
FR10: System shall compute relative velocity at TCA.
FR11: System shall categorise risk tier based on miss distance thresholds.

Risk Tier Logic:

* High: <1 km
* Medium: 1–5 km
* Low: 5–10 km

---

## 6.4 Asset Monitoring

FR12: User shall be able to designate a defended asset (NORAD ID).
FR13: System shall return conjunctions involving defended assets within selected time window.

---

## 6.5 Avoidance Simulation

FR14: System shall simulate a simple along-track delta-v manoeuvre.
FR15: System shall re-propagate defended asset post manoeuvre.
FR16: System shall compute updated miss distance.
FR17: System shall display improvement metrics.

The simulation shall be labelled as simplified and non-authoritative.

---

## 6.6 3D Visualisation

FR18: System shall render Earth in 3D.
FR19: System shall render both satellite trajectories.
FR20: System shall display closest approach point.
FR21: System shall include time slider for replay.
FR22: System shall allow toggle between pre- and post-manoeuvre state.

---

## 6.7 Congestion Index

FR23: System shall group objects into altitude bins (10 km bands).
FR24: System shall compute object density per band.
FR25: System shall compute conjunction rate per band.
FR26: System shall visualise results as chart.

---

# 7. Non-Functional Requirements

## 7.1 Performance

* Must handle minimum 5,000 objects.
* Conjunction screening per defended asset must complete within 60 seconds.
* Frontend rendering must remain interactive (>30 FPS in 3D view).

## 7.2 Reliability

* Ingestion failure must not crash API.
* Errors must be logged.

## 7.3 Transparency

* System must clearly state:

  * Public TLE only
  * Screening-grade system
  * No formal collision probability

---

# 8. Technical Architecture

## 8.1 Backend

Framework: FastAPI
Language: Python
Database: PostgreSQL
Orbit model: SGP4
Task scheduler: Background worker (Celery or internal scheduler)

Core Modules:

* ingest_service.py
* propagate_engine.py
* screening_engine.py
* avoidance_simulator.py
* congestion_analyser.py

---

## 8.2 API Endpoints

GET /catalog
GET /assets/{norad_id}/conjunctions
GET /conjunctions/{event_id}
POST /conjunctions/{event_id}/avoidance-sim
GET /congestion

---

## 8.3 Frontend

Framework: Next.js
Visualisation: Three.js
Charts: Lightweight charting library

Pages:

* Dashboard
* Asset View
* Conjunction Event View

---

# 9. User Flow

1. User opens dashboard.
2. User selects defended asset.
3. System displays upcoming conjunctions.
4. User selects event.
5. System displays 3D replay.
6. User clicks “Simulate Avoidance”.
7. System displays updated miss distance and orbit path.

---

# 10. Success Criteria (Hackathon)

* Real TLE data ingested.
* At least one live conjunction detected and visualised.
* Avoidance simulation produces measurable miss distance change.
* Congestion chart displays non-zero density.
* Demo is stable and reproducible.

---

# 11. Future Expansion Roadmap

Phase 2:

* Covariance integration
* Monte Carlo Pc modelling
* CDM compatibility

Phase 3:

* Constellation-scale distributed compute
* AI anomaly detection
* Congestion forecasting
* Manoeuvre coordination protocol layer

---

# 12. Key Risks

* TLE error propagation inaccuracies.
* Excessive compute time if screening not filtered properly.
* 3D performance issues.
* Overclaiming collision probability.

Mitigation:

* Restrict to LEO.
* Use altitude/inclination filtering.
* Explicit disclaimers.

---

# 13. Demo Script (For Hackathon)

1. “LEO is becoming congested.”
2. “We ingest public orbit data.”
3. Show live conjunction.
4. Replay closest approach.
5. Simulate avoidance.
6. Show congestion heatmap.
7. Explain long-term space traffic coordination vision.

---

If you want, I can now generate:

* A technical task breakdown by team member
* A 48-hour build sprint plan
* A pitch deck outline aligned to this PRD
* Or a GitHub-ready repo structure with initial files
