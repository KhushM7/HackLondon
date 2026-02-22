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
    ConjunctionSummary,
    CongestionBand,
    CongestionResponse,
    CustomSatelliteItem,
    CustomSatelliteListItem,
    CustomSatelliteRequest,
    CustomSatelliteResponse,
    OptimizeManeuverRequest,
    OptimizeManeuverResponse,
    TradeStudyEntry,
    TradeStudyResponse,
)
from .services.avoidance_simulator import AvoidanceSimulator
from .services.congestion_analyser import CongestionAnalyser
from .services.ingest_service import IngestService, TLESourceError
from .services.propagate_engine import PropagateEngine
from .services.screening_engine import ScreeningEngine
from .services.tle_validator import validate_tle

router = APIRouter()


@router.get("/catalog", response_model=list[CatalogItem])
def catalog(limit: int = Query(default=200, ge=1, le=5000), db: Session = Depends(get_session)):
    records = db.execute(select(TLERecord).limit(limit)).scalars().all()
    return records


@router.post("/ingest/refresh")
async def refresh_ingest(db: Session = Depends(get_session)):
    service = IngestService(db)
    try:
        count = await service.ingest_latest_tles()
        return {"ingested": count, "timestamp": datetime.utcnow()}
    except TLESourceError as exc:
        raise HTTPException(status_code=502, detail=f"Ingestion upstream error: {exc}")
    except Exception:
        raise HTTPException(status_code=500, detail="Ingestion failed")


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
    except TLESourceError as exc:
        raise HTTPException(status_code=502, detail=f"CelesTrak fetch failed: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CelesTrak fetch failed: {exc}")
    return record


@router.get("/catalog/custom-satellites", response_model=list[CustomSatelliteListItem])
def list_custom_satellites(db: Session = Depends(get_session)):
    """List all synthetic/custom satellites with conjunction event counts."""
    from sqlalchemy import func

    records = db.execute(
        select(TLERecord).where(TLERecord.is_synthetic == True)  # noqa: E712
    ).scalars().all()

    result = []
    for rec in records:
        count = db.execute(
            select(func.count()).where(
                (ConjunctionEvent.defended_norad_id == rec.norad_id)
                | (ConjunctionEvent.intruder_norad_id == rec.norad_id)
            )
        ).scalar() or 0
        result.append(
            CustomSatelliteListItem(
                norad_id=rec.norad_id,
                name=rec.name,
                inclination_deg=rec.inclination_deg,
                mean_motion=rec.mean_motion,
                source=rec.source,
                is_synthetic=rec.is_synthetic,
                created_at=rec.created_at,
                updated_at=rec.updated_at,
                conjunction_count=count,
            )
        )
    return result


@router.post("/catalog/custom-satellite", response_model=CustomSatelliteResponse, status_code=201)
def add_custom_satellite(payload: CustomSatelliteRequest, db: Session = Depends(get_session)):
    """Add a fully synthetic satellite from user-provided TLE. No external API calls."""
    line1 = payload.tle_line1.strip()
    line2 = payload.tle_line2.strip()
    sat_name = payload.name.strip()

    from sqlalchemy import func
    # Check duplicate name first to fail fast.
    existing_name = db.execute(
        select(TLERecord).where(func.lower(TLERecord.name) == sat_name.lower())
    ).scalar_one_or_none()
    if existing_name is not None:
        raise HTTPException(status_code=409, detail=f"Satellite with name '{sat_name}' already exists")

    # Validate TLE (checksums auto-corrected for hand-crafted TLEs)
    result = validate_tle(line1, line2)
    if not result.valid:
        raise HTTPException(status_code=422, detail={"tle_errors": result.errors})

    # Use checksum-corrected lines
    line1 = result.line1
    line2 = result.line2

    # Generate synthetic NORAD ID in 90000-99999 range
    max_id = db.execute(
        select(func.max(TLERecord.norad_id)).where(TLERecord.norad_id >= 90000, TLERecord.norad_id <= 99999)
    ).scalar()
    new_norad_id = (max_id + 1) if max_id is not None else 90000
    if new_norad_id > 99999:
        raise HTTPException(status_code=507, detail="Synthetic NORAD ID space exhausted (90000-99999)")

    now = datetime.utcnow()
    record = TLERecord(
        norad_id=new_norad_id,
        name=sat_name,
        line1=line1,
        line2=line2,
        inclination_deg=result.inclination_deg,
        mean_motion=result.mean_motion,
        eccentricity=result.eccentricity,
        raan_deg=result.raan_deg,
        arg_perigee_deg=result.arg_perigee_deg,
        bstar=result.bstar,
        epoch=result.epoch,
        source="custom",
        is_synthetic=True,
        mass_kg=payload.mass_kg,
        drag_area_m2=payload.drag_area_m2,
        radar_cross_section_m2=payload.radar_cross_section_m2,
        created_at=now,
        updated_at=now,
    )

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as exc:
        db.rollback()
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(status_code=409, detail=f"Satellite with name '{sat_name}' already exists")
        raise HTTPException(status_code=500, detail="Database error while storing satellite")

    # Run conjunction screening
    engine = ScreeningEngine(
        db,
        PropagateEngine(settings.propagation_horizon_days, settings.propagation_resolution_seconds),
    )
    conjunctions = engine.find_conjunctions(new_norad_id, days=settings.propagation_horizon_days)

    sat_item = CustomSatelliteItem(
        norad_id=record.norad_id,
        name=record.name,
        inclination_deg=record.inclination_deg,
        mean_motion=record.mean_motion,
        eccentricity=record.eccentricity,
        raan_deg=record.raan_deg,
        arg_perigee_deg=record.arg_perigee_deg,
        bstar=record.bstar,
        epoch=record.epoch,
        source=record.source,
        is_synthetic=record.is_synthetic,
        mass_kg=record.mass_kg,
        drag_area_m2=record.drag_area_m2,
        radar_cross_section_m2=record.radar_cross_section_m2,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )

    event_summaries = [
        ConjunctionSummary(
            id=e.id,
            intruder_norad_id=e.intruder_norad_id,
            tca_utc=e.tca_utc,
            miss_distance_km=e.miss_distance_km,
            risk_tier=e.risk_tier,
        )
        for e in conjunctions
    ]

    return CustomSatelliteResponse(
        satellite=sat_item,
        conjunctions_found=len(conjunctions),
        events=event_summaries,
    )


@router.delete("/catalog/custom-satellite/{norad_id}", status_code=200)
def delete_custom_satellite(norad_id: int, db: Session = Depends(get_session)):
    """Delete a synthetic satellite and cascade-delete its conjunction events."""
    from sqlalchemy import delete as sa_delete, or_

    record = db.execute(
        select(TLERecord).where(TLERecord.norad_id == norad_id)
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Satellite with NORAD ID {norad_id} not found")
    if not record.is_synthetic:
        raise HTTPException(status_code=403, detail="Cannot delete real catalog objects via this endpoint")

    # Cascade delete conjunction events
    db.execute(
        sa_delete(ConjunctionEvent).where(
            or_(
                ConjunctionEvent.defended_norad_id == norad_id,
                ConjunctionEvent.intruder_norad_id == norad_id,
            )
        )
    )
    db.delete(record)
    db.commit()
    return {"detail": f"Satellite {norad_id} and associated conjunction events deleted"}


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
