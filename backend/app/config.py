from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "OrbitGuard API"
    database_url: str = "sqlite:///./orbitguard.db"
    tle_source_url: str = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    ingestion_interval_hours: int = 6
    propagation_horizon_days: int = 7
    propagation_resolution_seconds: int = 60
    leo_max_altitude_km: float = 2000.0

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
