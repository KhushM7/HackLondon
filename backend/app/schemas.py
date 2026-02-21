from datetime import datetime

from pydantic import BaseModel, Field


class CatalogItem(BaseModel):
    norad_id: int
    name: str
    inclination_deg: float
    mean_motion: float
    updated_at: datetime


class ConjunctionResponse(BaseModel):
    id: int
    defended_norad_id: int
    intruder_norad_id: int
    tca_utc: datetime
    miss_distance_km: float
    relative_velocity_kms: float
    risk_tier: str


class ConjunctionDetail(ConjunctionResponse):
    pre_trajectory: list[dict]
    intruder_trajectory: list[dict]
    post_trajectory: list[dict] | None = None
    post_miss_distance_km: float | None = None


class AvoidanceRequest(BaseModel):
    delta_v_mps: float = Field(default=0.2, ge=0.01, le=5.0)


class AvoidanceResponse(BaseModel):
    event_id: int
    original_miss_distance_km: float
    updated_miss_distance_km: float
    delta_km: float
    note: str


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
    risk_tier: str | None = None


class AddSatelliteRequest(BaseModel):
    norad_id: int = Field(ge=1)
