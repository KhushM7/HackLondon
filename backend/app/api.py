from datetime import datetime, timedelta
import logging
from threading import Lock
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_session
from .models import AvoidancePlan, ConjunctionEvent, CustomSatelliteRegistry, TLERecord
from .schemas import (
    AddSatelliteRequest,
    AvoidancePlanSummary,
    AvoidanceRequest,
    AvoidanceResponse,
    CatalogItem,
    CatalogPosition,
    CatalogPositionsResponse,
    ConjunctionLink,
    ConjunctionDetail,
    ConjunctionResponse,
    CongestionBand,
    CongestionResponse,
    CustomSatelliteItem,
    CustomSatelliteAddQueued,
    CustomSatelliteAddStatus,
    CustomSatelliteListItem,
    CustomSatelliteRequest,
    CustomSatelliteResponse,
    OptimizeAvoidanceRequest,
    OrbitPathResponse,
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
logger = logging.getLogger(__name__)
_screening_locks_guard = Lock()
_screening_locks: dict[tuple[int, int], Lock] = {}
_custom_add_jobs_guard = Lock()
_custom_add_jobs: dict[str, dict] = {}


def _get_screening_lock(norad_id: int, days: int) -> Lock:
    key = (norad_id, days)
    with _screening_locks_guard:
        lock = _screening_locks.get(key)
        if lock is None:
            lock = Lock()
            _screening_locks[key] = lock
        return lock


def _set_custom_add_job(job_id: str, **changes) -> None:
    with _custom_add_jobs_guard:
        job = _custom_add_jobs.get(job_id)
        if job is None:
            return
        job.update(changes)
        job["updated_at"] = datetime.utcnow()


def _create_custom_add_job() -> str:
    job_id = uuid4().hex
    with _custom_add_jobs_guard:
        _custom_add_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress_pct": 0,
            "message": "Queued",
            "satellite": None,
            "conjunctions_found": None,
            "error": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    return job_id


def _get_custom_add_job(job_id: str) -> dict | None:
    with _custom_add_jobs_guard:
        job = _custom_add_jobs.get(job_id)
        if job is None:
            return None
        return dict(job)


def _name_map_for_events(db: Session, events: list[ConjunctionEvent]) -> dict[int, str]:
    ids: set[int] = set()
    for event in events:
        ids.add(event.defended_norad_id)
        ids.add(event.intruder_norad_id)
    if not ids:
        return {}
    rows = db.execute(
        select(TLERecord.norad_id, TLERecord.name).where(TLERecord.norad_id.in_(ids))
    ).all()
    return {norad_id: name for norad_id, name in rows}


def _asset_events_in_window(db: Session, norad_id: int, days: int) -> list[ConjunctionEvent]:
    now = datetime.utcnow()
    window_end = now + timedelta(days=days)
    return db.execute(
        select(ConjunctionEvent).where(
            or_(
                ConjunctionEvent.defended_norad_id == norad_id,
                ConjunctionEvent.intruder_norad_id == norad_id,
            ),
            ConjunctionEvent.tca_utc >= now,
            ConjunctionEvent.tca_utc <= window_end,
        )
    ).scalars().all()


def _conjunction_summary(event: ConjunctionEvent, name_map: dict[int, str]) -> dict:
    return {
        "id": event.id,
        "defended_norad_id": event.defended_norad_id,
        "defended_name": name_map.get(event.defended_norad_id),
        "intruder_norad_id": event.intruder_norad_id,
        "intruder_name": name_map.get(event.intruder_norad_id),
        "tca_utc": event.tca_utc,
        "miss_distance_km": event.miss_distance_km,
        "relative_velocity_kms": event.relative_velocity_kms,
        "risk_tier": event.risk_tier,
        "risk_level": event.risk_level,
        "pc_foster": event.pc_foster,
    }


def _conjunction_detail(event: ConjunctionEvent, name_map: dict[int, str]) -> dict:
    data = _conjunction_summary(event, name_map)
    data.update(
        {
            "pre_trajectory": event.pre_trajectory,
            "intruder_trajectory": event.intruder_trajectory,
            "post_trajectory": event.post_trajectory,
            "post_miss_distance_km": event.post_miss_distance_km,
            "propagation_method": event.propagation_method,
            "covariance_source": event.covariance_source,
            "tca_miss_vector_km": event.tca_miss_vector_km,
            "tca_relative_speed_kms": event.tca_relative_speed_kms,
            "b_dot_t_km": event.b_dot_t_km,
            "b_dot_r_km": event.b_dot_r_km,
            "b_magnitude_km": event.b_magnitude_km,
            "pc_chan": event.pc_chan,
            "pc_monte_carlo": event.pc_monte_carlo,
            "pc_monte_carlo_ci_low": event.pc_monte_carlo_ci_low,
            "pc_monte_carlo_ci_high": event.pc_monte_carlo_ci_high,
            "pc_method_used": event.pc_method_used,
            "hard_body_radius_km": event.hard_body_radius_km,
            "post_maneuver_delta_v_mps": event.post_maneuver_delta_v_mps,
            "post_maneuver_direction": event.post_maneuver_direction,
            "post_maneuver_burn_epoch": event.post_maneuver_burn_epoch,
            "post_maneuver_pc": event.post_maneuver_pc,
            "post_maneuver_miss_distance_km": event.post_maneuver_miss_distance_km,
            "post_maneuver_fuel_cost_kg": event.post_maneuver_fuel_cost_kg,
            "post_maneuver_trajectory": event.post_maneuver_trajectory,
        }
    )
    return data


def _component_priority(links: list[ConjunctionEvent]) -> tuple[int, float]:
    risk_order = {"High": 3, "Medium": 2, "Low": 1}
    max_risk = 0
    min_miss = float("inf")
    for event in links:
        max_risk = max(max_risk, risk_order.get(event.risk_tier, 0))
        min_miss = min(min_miss, event.miss_distance_km)
    return (max_risk, -min_miss)


def _custom_satellite_item_from_record(record: TLERecord) -> CustomSatelliteItem:
    return CustomSatelliteItem(
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


def _upsert_custom_satellite_registry(
    db: Session,
    record: TLERecord,
    *,
    status: str,
    conjunctions_found: int | None = None,
    last_error: str | None = None,
) -> None:
    row = db.execute(
        select(CustomSatelliteRegistry).where(CustomSatelliteRegistry.norad_id == record.norad_id)
    ).scalar_one_or_none()
    now = datetime.utcnow()
    if row is None:
        row = CustomSatelliteRegistry(
            norad_id=record.norad_id,
            name=record.name,
            status=status,
            conjunctions_found=conjunctions_found,
            last_error=last_error,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
    else:
        row.name = record.name
        row.status = status
        row.conjunctions_found = conjunctions_found
        row.last_error = last_error
        row.updated_at = now
    db.commit()


def _set_custom_satellite_registry_status(
    db: Session,
    norad_id: int,
    *,
    status: str,
    conjunctions_found: int | None = None,
    last_error: str | None = None,
) -> None:
    row = db.execute(
        select(CustomSatelliteRegistry).where(CustomSatelliteRegistry.norad_id == norad_id)
    ).scalar_one_or_none()
    if row is None:
        return
    row.status = status
    row.conjunctions_found = conjunctions_found
    row.last_error = last_error
    row.updated_at = datetime.utcnow()
    db.commit()


def _create_custom_satellite_record(db: Session, payload: CustomSatelliteRequest) -> TLERecord:
    from sqlalchemy import func

    line1 = payload.tle_line1.strip()
    line2 = payload.tle_line2.strip()
    sat_name = payload.name.strip()

    existing_name = db.execute(
        select(TLERecord).where(func.lower(TLERecord.name) == sat_name.lower())
    ).scalar_one_or_none()
    if existing_name is not None:
        raise HTTPException(status_code=409, detail=f"Satellite with name '{sat_name}' already exists")

    validation = validate_tle(line1, line2)
    if not validation.valid:
        raise HTTPException(status_code=422, detail={"tle_errors": validation.errors})

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
        line1=validation.line1,
        line2=validation.line2,
        inclination_deg=validation.inclination_deg,
        mean_motion=validation.mean_motion,
        eccentricity=validation.eccentricity,
        raan_deg=validation.raan_deg,
        arg_perigee_deg=validation.arg_perigee_deg,
        bstar=validation.bstar,
        epoch=validation.epoch,
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
    _upsert_custom_satellite_registry(db, record, status="stored", conjunctions_found=None, last_error=None)
    return record


def _select_satellite_ids_with_pairs(
    db: Session, limit: int, include_norad_id: int | None = None
) -> tuple[list[int], list[ConjunctionEvent]]:
    events = db.execute(select(ConjunctionEvent)).scalars().all()
    if not events:
        ids = db.execute(select(TLERecord.norad_id).limit(limit)).scalars().all()
        return list(ids), []

    graph: dict[int, set[int]] = {}
    events_by_node: dict[int, list[ConjunctionEvent]] = {}
    for event in events:
        a = event.defended_norad_id
        b = event.intruder_norad_id
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
        events_by_node.setdefault(a, []).append(event)
        events_by_node.setdefault(b, []).append(event)

    visited: set[int] = set()
    components: list[tuple[list[int], list[ConjunctionEvent]]] = []
    for node in graph.keys():
        if node in visited:
            continue
        stack = [node]
        comp_nodes: list[int] = []
        comp_events: list[ConjunctionEvent] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            comp_nodes.append(current)
            comp_events.extend(events_by_node.get(current, []))
            for neighbor in graph.get(current, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        # De-dup events in component
        unique_events = {e.id: e for e in comp_events}
        components.append((comp_nodes, list(unique_events.values())))

    components.sort(
        key=lambda item: _component_priority(item[1]),
        reverse=True,
    )

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_events: list[ConjunctionEvent] = []

    for comp_nodes, comp_events in components:
        if len(selected) + len(comp_nodes) > limit:
            continue
        for node in comp_nodes:
            if node not in selected_set:
                selected_set.add(node)
                selected.append(node)
        selected_events.extend(comp_events)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        graph_nodes = set(graph.keys())
        remaining = db.execute(
            select(TLERecord.norad_id)
            .where(
                ~TLERecord.norad_id.in_(selected_set),
                ~TLERecord.norad_id.in_(graph_nodes),
            )
            .limit(limit - len(selected))
        ).scalars().all()
        for norad_id in remaining:
            selected_set.add(norad_id)
            selected.append(norad_id)

    # Ensure a requested NORAD ID is present (e.g., a newly added custom satellite).
    if include_norad_id is not None and include_norad_id not in selected_set:
        exists = db.execute(
            select(TLERecord.norad_id).where(TLERecord.norad_id == include_norad_id)
        ).scalar_one_or_none()
        if exists is not None:
            if len(selected) >= limit and selected:
                dropped = selected.pop()
                selected_set.discard(dropped)
            selected.append(include_norad_id)
            selected_set.add(include_norad_id)

    # Only keep events whose endpoints are both selected
    selected_events = [
        event for event in events
        if event.defended_norad_id in selected_set and event.intruder_norad_id in selected_set
    ]

    return selected, selected_events


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
    refresh: bool = Query(default=False),
    db: Session = Depends(get_session),
):
    request_started = perf_counter()
    logger.info("API /assets/%s/conjunctions started (days=%s refresh=%s)", norad_id, days, refresh)

    if not refresh:
        cached = _asset_events_in_window(db, norad_id, days)
        if cached:
            ordered = sorted(cached, key=lambda x: x.miss_distance_km)
            name_map = _name_map_for_events(db, ordered)
            logger.info(
                "API /assets/%s/conjunctions cache hit: events=%s elapsed_s=%.1f",
                norad_id,
                len(ordered),
                perf_counter() - request_started,
            )
            return [_conjunction_summary(event, name_map) for event in ordered]

    lock = _get_screening_lock(norad_id, days)
    acquired = lock.acquire(blocking=False)

    if acquired:
        try:
            engine = ScreeningEngine(
                db,
                # Use requested window directly to avoid unnecessary propagation work.
                PropagateEngine(days, settings.propagation_resolution_seconds),
            )
            results = engine.find_conjunctions(norad_id, days=days)
            ordered = sorted(results, key=lambda x: x.miss_distance_km)
        finally:
            lock.release()
    else:
        wait_started = perf_counter()
        logger.info(
            "API /assets/%s/conjunctions waiting for in-flight screening (days=%s)",
            norad_id,
            days,
        )
        lock.acquire()
        lock.release()
        logger.info(
            "API /assets/%s/conjunctions joined in-flight screening after %.1fs",
            norad_id,
            perf_counter() - wait_started,
        )
        ordered = sorted(
            db.execute(
                select(ConjunctionEvent).where(ConjunctionEvent.defended_norad_id == norad_id)
            ).scalars().all(),
            key=lambda x: x.miss_distance_km,
        )

    # Return the same role-agnostic view as the cache path.
    ordered = sorted(_asset_events_in_window(db, norad_id, days), key=lambda x: x.miss_distance_km)
    name_map = _name_map_for_events(db, ordered)
    logger.info(
        "API /assets/%s/conjunctions finished: events=%s elapsed_s=%.1f",
        norad_id,
        len(ordered),
        perf_counter() - request_started,
    )
    return [_conjunction_summary(event, name_map) for event in ordered]


@router.get("/conjunctions/{event_id}", response_model=ConjunctionDetail)
def conjunction_detail(event_id: int, db: Session = Depends(get_session)):
    event = db.execute(select(ConjunctionEvent).where(ConjunctionEvent.id == event_id)).scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail="Conjunction event not found")
    name_map = _name_map_for_events(db, [event])
    return _conjunction_detail(event, name_map)


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


def _run_avoidance_optimization(
    norad_id: int,
    plan_id: int,
    max_delta_v_mps: float,
    burn_window_hours: float,
    weight_miss_distance: float,
    weight_delta_v: float,
    top_n_events: int,
) -> None:
    """Background task: run avoidance optimization with its own DB session."""
    from .db import SessionLocal
    db = SessionLocal()
    try:
        simulator = AvoidanceSimulator(db)
        simulator.run_full_optimization(
            norad_id,
            plan_id=plan_id,
            max_delta_v_mps=max_delta_v_mps,
            burn_window_hours=burn_window_hours,
            weight_miss_distance=weight_miss_distance,
            weight_delta_v=weight_delta_v,
            top_n_events=top_n_events,
        )
    except Exception as exc:
        logger.error("Background optimization failed for %s: %s", norad_id, exc)
    finally:
        db.close()


def _run_custom_satellite_screening(norad_id: int, days: int) -> None:
    """Background task: screen conjunctions for a newly added custom satellite."""
    from .db import SessionLocal

    db = SessionLocal()
    lock = _get_screening_lock(norad_id, days)
    acquired = lock.acquire(blocking=False)
    started = perf_counter()
    try:
        if not acquired:
            logger.info(
                "Custom screening skipped: norad_id=%s days=%s reason=in_flight",
                norad_id,
                days,
            )
            return

        logger.info("Custom screening started: norad_id=%s days=%s", norad_id, days)
        _set_custom_satellite_registry_status(
            db,
            norad_id,
            status="screening",
            conjunctions_found=None,
            last_error=None,
        )
        engine = ScreeningEngine(
            db,
            PropagateEngine(days, settings.propagation_resolution_seconds),
        )
        events = engine.find_conjunctions(norad_id, days=days)
        _set_custom_satellite_registry_status(
            db,
            norad_id,
            status="completed",
            conjunctions_found=len(events),
            last_error=None,
        )
        logger.info(
            "Custom screening finished: norad_id=%s days=%s events=%s elapsed_s=%.1f",
            norad_id,
            days,
            len(events),
            perf_counter() - started,
        )
    except Exception:
        _set_custom_satellite_registry_status(
            db,
            norad_id,
            status="failed",
            conjunctions_found=None,
            last_error="screening_failed",
        )
        logger.exception("Custom screening failed: norad_id=%s days=%s", norad_id, days)
    finally:
        if acquired:
            lock.release()
        db.close()


@router.post("/assets/{norad_id}/avoidance/optimize", response_model=AvoidancePlanSummary, status_code=202)
def start_avoidance_optimization(
    norad_id: int,
    payload: OptimizeAvoidanceRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
):
    """Start async optimization for top-risk conjunctions of an asset.

    Returns immediately with a plan in 'running' status. Poll GET /assets/{id}/avoidance/plan
    for results.
    """
    # Verify asset exists
    asset = db.execute(select(TLERecord).where(TLERecord.norad_id == norad_id)).scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset {norad_id} not found")

    # Create plan now and let background worker fill it in.
    plan = AvoidancePlan(
        asset_norad_id=norad_id,
        status="running",
        progress_stage="queued",
        progress_done=0,
        progress_total=None,
        progress_message="Queued for optimization",
        heartbeat_at=datetime.utcnow(),
        optimizer_version="v1",
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    logger.info(
        "API /assets/%s/avoidance/optimize queued: plan_id=%s max_dv=%.2f window_h=%.1f top_n=%s",
        norad_id,
        plan.id,
        payload.max_delta_v_mps,
        payload.burn_window_hours,
        payload.top_n_events,
    )

    # Launch background optimization
    background_tasks.add_task(
        _run_avoidance_optimization,
        norad_id,
        plan.id,
        payload.max_delta_v_mps,
        payload.burn_window_hours,
        payload.weight_miss_distance,
        payload.weight_delta_v,
        payload.top_n_events,
    )

    return AvoidancePlanSummary(
        id=plan.id,
        asset_norad_id=plan.asset_norad_id,
        status=plan.status,
        progress_stage=plan.progress_stage,
        progress_done=plan.progress_done,
        progress_total=plan.progress_total,
        progress_message=plan.progress_message,
        heartbeat_at=plan.heartbeat_at,
        created_at=plan.created_at,
    )


@router.get("/assets/{norad_id}/avoidance/plan", response_model=AvoidancePlanSummary)
def get_avoidance_plan(norad_id: int, db: Session = Depends(get_session)):
    """Get the latest computed avoidance plan for an asset."""
    plan = db.execute(
        select(AvoidancePlan)
        .where(AvoidancePlan.asset_norad_id == norad_id)
        .order_by(AvoidancePlan.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    if plan is None:
        raise HTTPException(status_code=404, detail=f"No avoidance plan found for asset {norad_id}")

    return AvoidancePlanSummary(
        id=plan.id,
        asset_norad_id=plan.asset_norad_id,
        status=plan.status,
        event_id=plan.event_id,
        burn_direction=plan.burn_direction,
        burn_dv_mps=plan.burn_dv_mps,
        burn_rtn_vector=plan.burn_rtn_vector,
        burn_epoch=plan.burn_epoch,
        pre_miss_distance_km=plan.pre_miss_distance_km,
        post_miss_distance_km=plan.post_miss_distance_km,
        pre_pc=plan.pre_pc,
        post_pc=plan.post_pc,
        fuel_cost_kg=plan.fuel_cost_kg,
        current_path=plan.current_path,
        deviated_path=plan.deviated_path,
        candidates_evaluated=plan.candidates_evaluated,
        optimization_elapsed_s=plan.optimization_elapsed_s,
        progress_stage=plan.progress_stage,
        progress_done=plan.progress_done,
        progress_total=plan.progress_total,
        progress_message=plan.progress_message,
        heartbeat_at=plan.heartbeat_at,
        error_message=plan.error_message,
        created_at=plan.created_at,
        completed_at=plan.completed_at,
    )


@router.post("/avoidance/cancel-all")
def cancel_all_avoidance_plans(db: Session = Depends(get_session)):
    """Cancel all running/pending avoidance optimizations."""
    now = datetime.utcnow()
    plans = db.execute(
        select(AvoidancePlan).where(AvoidancePlan.status.in_(["pending", "running"]))
    ).scalars().all()

    for plan in plans:
        plan.status = "cancelled"
        plan.completed_at = now
        plan.progress_stage = "cancelled"
        plan.progress_done = 1
        plan.progress_total = 1
        plan.progress_message = "Cancelled by user"
        plan.heartbeat_at = now
        if not plan.error_message:
            plan.error_message = "Cancelled by user"

    db.commit()
    logger.info("Cancelled %s avoidance plan(s)", len(plans))
    return {"cancelled": len(plans), "timestamp": now}


@router.post("/catalog/satellite", response_model=CatalogItem, status_code=201)
async def add_satellite(payload: AddSatelliteRequest, db: Session = Depends(get_session)):
    existing = db.execute(
        select(TLERecord).where(TLERecord.norad_id == payload.norad_id)
    ).scalar_one_or_none()
    if existing is not None:
        return existing

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


def _run_custom_satellite_add_job(job_id: str, payload_data: dict) -> None:
    from .db import SessionLocal

    db = SessionLocal()
    record: TLERecord | None = None
    try:
        _set_custom_add_job(
            job_id,
            status="running",
            stage="validating",
            progress_pct=8,
            message="Validating TLE",
        )
        payload = CustomSatelliteRequest(**payload_data)

        _set_custom_add_job(
            job_id,
            stage="storing",
            progress_pct=22,
            message="Saving satellite",
        )
        record = _create_custom_satellite_record(db, payload)

        screen_days = max(1, min(settings.propagation_horizon_days, settings.custom_satellite_screen_days))
        _set_custom_satellite_registry_status(
            db,
            record.norad_id,
            status="screening",
            conjunctions_found=None,
            last_error=None,
        )
        _set_custom_add_job(
            job_id,
            stage="screening",
            progress_pct=35,
            message="Running conjunction screening",
        )

        def _progress(processed: int, total: int, hits: int) -> None:
            if total <= 0:
                pct = 95
                msg = "Screening complete"
            else:
                frac = max(0.0, min(1.0, processed / total))
                pct = min(95, 35 + int(frac * 60))
                msg = f"Screening {processed}/{total} candidates, {hits} hits"
            _set_custom_add_job(
                job_id,
                stage="screening",
                progress_pct=pct,
                message=msg,
            )

        lock = _get_screening_lock(record.norad_id, screen_days)
        lock.acquire()
        try:
            engine = ScreeningEngine(
                db,
                PropagateEngine(screen_days, settings.propagation_resolution_seconds),
            )
            events = engine.find_conjunctions(
                record.norad_id,
                days=screen_days,
                progress_callback=_progress,
            )
        finally:
            lock.release()

        sat_item = _custom_satellite_item_from_record(record)
        _set_custom_satellite_registry_status(
            db,
            record.norad_id,
            status="completed",
            conjunctions_found=len(events),
            last_error=None,
        )
        _set_custom_add_job(
            job_id,
            status="completed",
            stage="completed",
            progress_pct=100,
            message="Satellite added successfully",
            satellite=sat_item.model_dump(mode="json"),
            conjunctions_found=len(events),
            error=None,
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        if record is not None:
            _set_custom_satellite_registry_status(
                db,
                record.norad_id,
                status="failed",
                conjunctions_found=None,
                last_error=detail,
            )
        _set_custom_add_job(
            job_id,
            status="failed",
            stage="failed",
            progress_pct=100,
            message="Add satellite failed",
            error=detail,
        )
    except Exception as exc:
        if record is not None:
            _set_custom_satellite_registry_status(
                db,
                record.norad_id,
                status="failed",
                conjunctions_found=None,
                last_error=str(exc),
            )
        _set_custom_add_job(
            job_id,
            status="failed",
            stage="failed",
            progress_pct=100,
            message="Add satellite failed",
            error=str(exc),
        )
        logger.exception("Custom satellite add job failed: job_id=%s", job_id)
    finally:
        db.close()


@router.post("/catalog/custom-satellite/submit", response_model=CustomSatelliteAddQueued, status_code=202)
def submit_custom_satellite(
    payload: CustomSatelliteRequest,
    background_tasks: BackgroundTasks,
):
    job_id = _create_custom_add_job()
    background_tasks.add_task(
        _run_custom_satellite_add_job,
        job_id,
        payload.model_dump(by_alias=True),
    )
    return CustomSatelliteAddQueued(
        job_id=job_id,
        status="queued",
        stage="queued",
        progress_pct=0,
        message="Queued",
    )


@router.get("/catalog/custom-satellite/jobs/{job_id}", response_model=CustomSatelliteAddStatus)
def custom_satellite_job_status(job_id: str):
    job = _get_custom_add_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    satellite = None
    if isinstance(job.get("satellite"), dict):
        satellite = CustomSatelliteItem(**job["satellite"])
    return CustomSatelliteAddStatus(
        job_id=job["job_id"],
        status=job["status"],
        stage=job["stage"],
        progress_pct=int(job["progress_pct"]),
        message=job.get("message"),
        satellite=satellite,
        conjunctions_found=job.get("conjunctions_found"),
        error=job.get("error"),
    )


@router.post("/catalog/custom-satellite", response_model=CustomSatelliteResponse, status_code=201)
def add_custom_satellite(
    payload: CustomSatelliteRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
):
    """Add a fully synthetic satellite from user-provided TLE. No external API calls."""
    endpoint_started = perf_counter()
    record = _create_custom_satellite_record(db, payload)

    screen_days = max(1, min(settings.propagation_horizon_days, settings.custom_satellite_screen_days))
    background_tasks.add_task(_run_custom_satellite_screening, record.norad_id, screen_days)
    logger.info(
        "API /catalog/custom-satellite queued screening: norad_id=%s days=%s",
        record.norad_id,
        screen_days,
    )

    sat_item = _custom_satellite_item_from_record(record)

    response = CustomSatelliteResponse(
        satellite=sat_item,
        conjunctions_found=0,
        events=[],
    )
    logger.info(
        "API /catalog/custom-satellite finished: norad_id=%s total_elapsed_s=%.1f",
        record.norad_id,
        perf_counter() - endpoint_started,
    )
    return response


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
    db.execute(
        sa_delete(CustomSatelliteRegistry).where(CustomSatelliteRegistry.norad_id == norad_id)
    )
    db.delete(record)
    db.commit()
    return {"detail": f"Satellite {norad_id} and associated conjunction events deleted"}


@router.get("/catalog/positions", response_model=CatalogPositionsResponse)
def catalog_positions(
    limit: int = Query(default=500, ge=1, le=5000),
    focus_norad_id: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_session),
):
    from sgp4.api import Satrec, jday as sgp4_jday

    now = datetime.utcnow()
    jd, fr = sgp4_jday(now.year, now.month, now.day, now.hour, now.minute, now.second + now.microsecond / 1e6)

    selected_ids, selected_events = _select_satellite_ids_with_pairs(
        db,
        limit,
        include_norad_id=focus_norad_id,
    )
    if not selected_ids:
        return CatalogPositionsResponse(satellites=[], links=[])

    records = db.execute(
        select(TLERecord).where(TLERecord.norad_id.in_(selected_ids))
    ).scalars().all()

    risk_order = {"High": 3, "Medium": 2, "Low": 1}
    risk_map: dict[int, str] = {}
    for event in selected_events:
        tier = event.risk_tier
        for norad_id in (event.defended_norad_id, event.intruder_norad_id):
            if risk_order.get(tier, 0) > risk_order.get(risk_map.get(norad_id, ""), 0):
                risk_map[norad_id] = tier

    result: list[CatalogPosition] = []
    for record in records:
        try:
            sat = Satrec.twoline2rv(record.line1, record.line2)
            err, position, velocity = sat.sgp4(jd, fr)
            if err == 0:
                result.append(
                    CatalogPosition(
                        norad_id=record.norad_id,
                        name=record.name,
                        position_km=[float(v) for v in position],
                        velocity_kms=[float(v) for v in velocity],
                        risk_tier=risk_map.get(record.norad_id),
                    )
                )
        except Exception:
            pass

    links = [
        ConjunctionLink(
            event_id=event.id,
            defended_norad_id=event.defended_norad_id,
            intruder_norad_id=event.intruder_norad_id,
            risk_tier=event.risk_tier,
            miss_distance_km=event.miss_distance_km,
        )
        for event in selected_events
    ]

    return CatalogPositionsResponse(satellites=result, links=links)


@router.get("/congestion", response_model=CongestionResponse)
def congestion(db: Session = Depends(get_session)):
    analyser = CongestionAnalyser(db)
    bands = [CongestionBand(**band) for band in analyser.compute()]
    return CongestionResponse(generated_at=datetime.utcnow(), bands=bands)


@router.get("/catalog/orbit/{norad_id}", response_model=OrbitPathResponse)
def catalog_orbit(
    norad_id: int,
    orbits: float = Query(default=1.0, ge=0.5, le=4.0),
    db: Session = Depends(get_session),
):
    from sgp4.api import Satrec, jday as sgp4_jday

    record = db.execute(select(TLERecord).where(TLERecord.norad_id == norad_id)).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Satellite {norad_id} not found")

    mean_motion = record.mean_motion or 0.0
    period_seconds = 86400.0 / mean_motion if mean_motion > 0 else 5400.0
    period_seconds = max(1800.0, min(period_seconds, 21600.0))
    total_seconds = period_seconds * orbits
    step_seconds = 120.0

    now = datetime.utcnow()
    sat = Satrec.twoline2rv(record.line1, record.line2)
    positions: list[list[float]] = []
    t = 0.0
    while t <= total_seconds:
        dt = now + timedelta(seconds=t)
        jd, fr = sgp4_jday(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second + dt.microsecond / 1e6,
        )
        err, pos, _vel = sat.sgp4(jd, fr)
        if err == 0:
            positions.append([float(v) for v in pos])
        t += step_seconds

    return OrbitPathResponse(norad_id=norad_id, positions_km=positions)
