from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_session
from .models import ConjunctionEvent, TLERecord
from .schemas import (
    AddSatelliteRequest,
    AvoidanceRequest,
    AvoidanceResponse,
    CatalogItem,
    CatalogPosition,
    ConjunctionDetail,
    ConjunctionResponse,
    CongestionBand,
    CongestionResponse,
    OptimizeManeuverRequest,
    OptimizeManeuverResponse,
    TradeStudyEntry,
    TradeStudyResponse,
)
from .services.avoidance_simulator import AvoidanceSimulator
from .services.congestion_analyser import CongestionAnalyser
from .services.ingest_service import IngestService
from .services.propagate_engine import PropagateEngine
from .services.screening_engine import ScreeningEngine

router = APIRouter()


@router.get("/catalog", response_model=list[CatalogItem])
def catalog(limit: int = Query(default=200, ge=1, le=5000), db: Session = Depends(get_session)):
    records = db.execute(select(TLERecord).limit(limit)).scalars().all()
    return records


@router.post("/ingest/refresh")
async def refresh_ingest(db: Session = Depends(get_session)):
    service = IngestService(db)
    count = await service.ingest_latest_tles()
    return {"ingested": count, "timestamp": datetime.utcnow()}


@router.get("/assets/{norad_id}/conjunctions", response_model=list[ConjunctionResponse])
def asset_conjunctions(
    norad_id: int,
    days: int = Query(default=3, ge=1, le=settings.propagation_horizon_days),
    db: Session = Depends(get_session),
):
    engine = ScreeningEngine(
        db,
        PropagateEngine(settings.propagation_horizon_days, settings.propagation_resolution_seconds),
    )
    results = engine.find_conjunctions(norad_id, days=days)
    return sorted(results, key=lambda x: x.miss_distance_km)


@router.get("/conjunctions/{event_id}", response_model=ConjunctionDetail)
def conjunction_detail(event_id: int, db: Session = Depends(get_session)):
    event = db.execute(select(ConjunctionEvent).where(ConjunctionEvent.id == event_id)).scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail="Conjunction event not found")
    return event


@router.post("/conjunctions/{event_id}/avoidance-sim", response_model=AvoidanceResponse)
def avoidance_sim(event_id: int, payload: AvoidanceRequest, db: Session = Depends(get_session)):
    event = db.execute(select(ConjunctionEvent).where(ConjunctionEvent.id == event_id)).scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail="Conjunction event not found")

    simulator = AvoidanceSimulator(db)
    updated_miss_distance, _ = simulator.simulate(event, payload.delta_v_mps, payload.direction)
    return AvoidanceResponse(
        event_id=event.id,
        original_miss_distance_km=event.miss_distance_km,
        updated_miss_distance_km=updated_miss_distance,
        delta_km=updated_miss_distance - event.miss_distance_km,
        note="High-fidelity RTN maneuver simulation with Pc computation.",
        post_pc=event.post_maneuver_pc,
        fuel_cost_kg=event.post_maneuver_fuel_cost_kg,
        direction=payload.direction,
    )


@router.post("/events/{event_id}/optimize-maneuver", response_model=OptimizeManeuverResponse)
def optimize_maneuver_endpoint(
    event_id: int,
    payload: OptimizeManeuverRequest,
    db: Session = Depends(get_session),
):
    """Find minimum delta-v burn to achieve target Pc threshold."""
    event = db.execute(select(ConjunctionEvent).where(ConjunctionEvent.id == event_id)).scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail="Conjunction event not found")

    simulator = AvoidanceSimulator(db)
    try:
        result = simulator.run_optimize(
            event,
            target_pc=payload.target_pc,
            direction=payload.direction,
            max_dv_mps=payload.max_dv_mps,
            burn_lead_time_hours=payload.burn_lead_time_hours,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return OptimizeManeuverResponse(**result)


@router.get("/events/{event_id}/trade-study", response_model=TradeStudyResponse)
def trade_study_endpoint(event_id: int, db: Session = Depends(get_session)):
    """Return maneuver trade table across all 6 RTN directions."""
    event = db.execute(select(ConjunctionEvent).where(ConjunctionEvent.id == event_id)).scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail="Conjunction event not found")

    simulator = AvoidanceSimulator(db)
    try:
        entries = simulator.run_trade_study(event)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return TradeStudyResponse(
        event_id=event.id,
        entries=[TradeStudyEntry(**e) for e in entries],
    )


@router.post("/catalog/satellite", response_model=CatalogItem, status_code=201)
async def add_satellite(payload: AddSatelliteRequest, db: Session = Depends(get_session)):
    service = IngestService(db)
    try:
        record = await service.ingest_satellite_by_norad_id(payload.norad_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CelesTrak fetch failed: {exc}")
    return record


@router.get("/catalog/positions", response_model=list[CatalogPosition])
def catalog_positions(limit: int = Query(default=500, ge=1, le=2000), db: Session = Depends(get_session)):
    from sgp4.api import Satrec, jday as sgp4_jday

    now = datetime.utcnow()
    jd, fr = sgp4_jday(now.year, now.month, now.day, now.hour, now.minute, now.second + now.microsecond / 1e6)

    records = db.execute(select(TLERecord).limit(limit)).scalars().all()

    risk_order = {"High": 3, "Medium": 2, "Low": 1}
    rows = db.execute(select(ConjunctionEvent.defended_norad_id, ConjunctionEvent.risk_tier)).all()
    risk_map: dict[int, str] = {}
    for norad_id, tier in rows:
        if risk_order.get(tier, 0) > risk_order.get(risk_map.get(norad_id, ""), 0):
            risk_map[norad_id] = tier

    result: list[CatalogPosition] = []
    for record in records:
        try:
            sat = Satrec.twoline2rv(record.line1, record.line2)
            err, position, _ = sat.sgp4(jd, fr)
            if err == 0:
                result.append(
                    CatalogPosition(
                        norad_id=record.norad_id,
                        name=record.name,
                        position_km=[float(v) for v in position],
                        risk_tier=risk_map.get(record.norad_id),
                    )
                )
        except Exception:
            pass
    return result


@router.get("/congestion", response_model=CongestionResponse)
def congestion(db: Session = Depends(get_session)):
    analyser = CongestionAnalyser(db)
    bands = [CongestionBand(**band) for band in analyser.compute()]
    return CongestionResponse(generated_at=datetime.utcnow(), bands=bands)
