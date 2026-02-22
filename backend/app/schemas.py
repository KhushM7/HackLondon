from datetime import datetime

from pydantic import BaseModel, Field


class CatalogItem(BaseModel):
    norad_id: int
    name: str
    inclination_deg: float
    mean_motion: float
    updated_at: datetime


class ConjunctionLink(BaseModel):
    event_id: int
    defended_norad_id: int
    intruder_norad_id: int
    risk_tier: str
    miss_distance_km: float


class ConjunctionResponse(BaseModel):
    id: int
    defended_norad_id: int
    defended_name: str | None = None
    intruder_norad_id: int
    intruder_name: str | None = None
    tca_utc: datetime
    miss_distance_km: float
    relative_velocity_kms: float
    risk_tier: str
    risk_level: str | None = None
    pc_foster: float | None = None


class ConjunctionDetail(ConjunctionResponse):
    pre_trajectory: list[dict]
    intruder_trajectory: list[dict]
    post_trajectory: list[dict] | None = None
    post_miss_distance_km: float | None = None

    # High-fidelity fields
    propagation_method: str | None = None
    covariance_source: str | None = None
    tca_miss_vector_km: list[float] | None = None
    tca_relative_speed_kms: float | None = None

    # B-plane
    b_dot_t_km: float | None = None
    b_dot_r_km: float | None = None
    b_magnitude_km: float | None = None

    # Probability of Collision
    pc_chan: float | None = None
    pc_monte_carlo: float | None = None
    pc_monte_carlo_ci_low: float | None = None
    pc_monte_carlo_ci_high: float | None = None
    pc_method_used: str | None = None

    # Covariance
    hard_body_radius_km: float | None = None

    # Post-maneuver
    post_maneuver_delta_v_mps: float | None = None
    post_maneuver_direction: str | None = None
    post_maneuver_burn_epoch: datetime | None = None
    post_maneuver_pc: float | None = None
    post_maneuver_miss_distance_km: float | None = None
    post_maneuver_fuel_cost_kg: float | None = None
    post_maneuver_trajectory: list[dict] | None = None


class AvoidanceRequest(BaseModel):
    delta_v_mps: float = Field(default=0.2, ge=0.01, le=5.0)
    direction: str = Field(default="prograde", description="RTN direction: prograde, retrograde, radial_plus, radial_minus, normal_plus, normal_minus")


class AvoidanceResponse(BaseModel):
    event_id: int
    original_miss_distance_km: float
    updated_miss_distance_km: float
    delta_km: float
    note: str
    post_pc: float | None = None
    fuel_cost_kg: float | None = None
    direction: str | None = None


class OptimizeManeuverRequest(BaseModel):
    target_pc: float = Field(default=1e-4, gt=0, lt=1.0, description="Target maximum Pc")
    direction: str = Field(default="prograde", description="RTN burn direction")
    max_dv_mps: float = Field(default=10.0, ge=0.01, le=50.0, description="Maximum delta-v (m/s)")
    burn_lead_time_hours: float = Field(default=24.0, ge=0.5, le=168.0, description="Hours before TCA to burn")


class OptimizeManeuverResponse(BaseModel):
    event_id: int
    optimal_delta_v_mps: float
    direction: str
    burn_epoch: datetime
    pre_pc: float
    post_pc: float
    post_miss_distance_km: float
    fuel_cost_kg: float


class TradeStudyEntry(BaseModel):
    direction: str
    delta_v_mps: float
    post_pc: float
    post_miss_distance_km: float
    fuel_cost_kg: float


class TradeStudyResponse(BaseModel):
    event_id: int
    entries: list[TradeStudyEntry]


class CongestionBand(BaseModel):
    altitude_band_km: str
    object_count: int
    conjunction_rate: float


class CongestionResponse(BaseModel):
    generated_at: datetime
    bands: list[CongestionBand]


class CatalogPosition(BaseModel):
    norad_id: int
    name: str
    position_km: list[float]
    velocity_kms: list[float]
    risk_tier: str | None = None


class CatalogPositionsResponse(BaseModel):
    satellites: list[CatalogPosition]
    links: list[ConjunctionLink]


class OrbitPathResponse(BaseModel):
    norad_id: int
    positions_km: list[list[float]]


class AddSatelliteRequest(BaseModel):
    norad_id: int = Field(ge=1)


class CustomSatelliteRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    tle_line1: str = Field(alias="line1")
    tle_line2: str = Field(alias="line2")
    mass_kg: float | None = None
    drag_area_m2: float | None = None
    radar_cross_section_m2: float | None = None
    is_synthetic: bool = True

    model_config = {"populate_by_name": True}


class CustomSatelliteItem(BaseModel):
    norad_id: int
    name: str
    inclination_deg: float
    mean_motion: float
    eccentricity: float | None = None
    raan_deg: float | None = None
    arg_perigee_deg: float | None = None
    bstar: float | None = None
    epoch: datetime | None = None
    source: str | None = None
    is_synthetic: bool | None = None
    mass_kg: float | None = None
    drag_area_m2: float | None = None
    radar_cross_section_m2: float | None = None
    created_at: datetime | None = None
    updated_at: datetime


class ConjunctionSummary(BaseModel):
    id: int
    intruder_norad_id: int
    tca_utc: datetime
    miss_distance_km: float
    risk_tier: str


class CustomSatelliteResponse(BaseModel):
    satellite: CustomSatelliteItem
    conjunctions_found: int
    events: list[ConjunctionSummary]


class CustomSatelliteAddQueued(BaseModel):
    job_id: str
    status: str
    stage: str
    progress_pct: int
    message: str | None = None


class CustomSatelliteAddStatus(BaseModel):
    job_id: str
    status: str
    stage: str
    progress_pct: int
    message: str | None = None
    satellite: CustomSatelliteItem | None = None
    conjunctions_found: int | None = None
    error: str | None = None


class CustomSatelliteListItem(BaseModel):
    norad_id: int
    name: str
    inclination_deg: float
    mean_motion: float
    source: str | None = None
    is_synthetic: bool | None = None
    created_at: datetime | None = None
    updated_at: datetime
    conjunction_count: int = 0


class OptimizeAvoidanceRequest(BaseModel):
    max_delta_v_mps: float = Field(default=5.0, ge=0.01, le=50.0, description="Maximum delta-v budget (m/s)")
    burn_window_hours: float = Field(default=48.0, ge=1.0, le=168.0, description="Allowed burn window before TCA (hours)")
    weight_miss_distance: float = Field(default=1.0, ge=0.0, le=10.0, description="Objective weight for miss distance maximization")
    weight_delta_v: float = Field(default=0.3, ge=0.0, le=10.0, description="Objective weight for delta-v minimization")
    top_n_events: int = Field(default=3, ge=1, le=20, description="Number of top-risk events to optimize against")


class AvoidancePlanSummary(BaseModel):
    id: int
    asset_norad_id: int
    status: str
    event_id: int | None = None
    burn_direction: str | None = None
    burn_dv_mps: float | None = None
    burn_rtn_vector: list[float] | None = None
    burn_epoch: datetime | None = None
    pre_miss_distance_km: float | None = None
    post_miss_distance_km: float | None = None
    pre_pc: float | None = None
    post_pc: float | None = None
    fuel_cost_kg: float | None = None
    current_path: list[dict] | None = None
    deviated_path: list[dict] | None = None
    candidates_evaluated: int | None = None
    optimization_elapsed_s: float | None = None
    progress_stage: str | None = None
    progress_done: int | None = None
    progress_total: int | None = None
    progress_message: str | None = None
    heartbeat_at: datetime | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
