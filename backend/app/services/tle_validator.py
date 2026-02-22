"""TLE validation using format checks and SGP4 parsing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sgp4.api import Satrec, jday


@dataclass
class TLEValidationResult:
    valid: bool
    errors: list[str]
    sat: Satrec | None = None
    line1: str = ""
    line2: str = ""
    inclination_deg: float = 0.0
    raan_deg: float = 0.0
    eccentricity: float = 0.0
    arg_perigee_deg: float = 0.0
    mean_motion: float = 0.0
    bstar: float = 0.0
    epoch: datetime | None = None


def _tle_checksum(line: str) -> int:
    """Compute TLE modulo-10 checksum for columns 1-68."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def _fix_checksum(line: str) -> str:
    """Replace the last character of a TLE line with the correct checksum."""
    return line[:68] + str(_tle_checksum(line))


def validate_tle(line1: str, line2: str) -> TLEValidationResult:
    """Validate TLE lines and return parsed orbital elements or errors.
    
    Checksums are auto-corrected since users craft these by hand.
    """
    errors: list[str] = []

    # Basic length check
    if len(line1) < 69:
        errors.append(f"TLE line 1 too short: expected 69 chars, got {len(line1)}")
    if len(line2) < 69:
        errors.append(f"TLE line 2 too short: expected 69 chars, got {len(line2)}")
    if errors:
        return TLEValidationResult(valid=False, errors=errors)

    # Line number check
    if line1[0] != "1":
        errors.append("TLE line 1 must start with '1'")
    if line2[0] != "2":
        errors.append("TLE line 2 must start with '2'")

    if errors:
        return TLEValidationResult(valid=False, errors=errors)

    # Auto-correct checksums for user-crafted TLEs
    line1 = _fix_checksum(line1)
    line2 = _fix_checksum(line2)

    # Catalog number match
    try:
        cat1 = int(line1[2:7])
        cat2 = int(line2[2:7])
        if cat1 != cat2:
            errors.append(
                f"Catalog number mismatch: line 1 has {cat1}, line 2 has {cat2}"
            )
    except ValueError:
        errors.append("Cannot parse catalog number from TLE lines")

    if errors:
        return TLEValidationResult(valid=False, errors=errors)

    # SGP4 parsing
    try:
        sat = Satrec.twoline2rv(line1, line2)
    except Exception as exc:
        errors.append(f"SGP4 parsing failed: {exc}")
        return TLEValidationResult(valid=False, errors=errors)

    # Propagate 1 step to confirm valid state vector
    now = datetime.utcnow()
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute,
                  now.second + now.microsecond / 1e6)
    err, position, velocity = sat.sgp4(jd, fr)
    if err != 0:
        error_codes = {
            1: "mean elements (eccentricity >= 1.0 or < -0.001, or a < 0.95)",
            2: "mean motion less than 0.0",
            3: "perturbation elements (eccentricity < 0.0 or > 1.0)",
            4: "semi-latus rectum < 0.0",
            5: "epoch elements are sub-orbital (mrt < 1.0)",
            6: "satellite has decayed",
        }
        msg = error_codes.get(err, f"unknown error code {err}")
        errors.append(f"SGP4 propagation error (code {err}): {msg}")
        return TLEValidationResult(valid=False, errors=errors)

    # Parse epoch from TLE
    epoch_year = int(line1[18:20])
    epoch_day = float(line1[20:32])
    full_year = 2000 + epoch_year if epoch_year < 57 else 1900 + epoch_year
    epoch_dt = datetime(full_year, 1, 1) + __import__("datetime").timedelta(days=epoch_day - 1)

    return TLEValidationResult(
        valid=True,
        errors=[],
        sat=sat,
        line1=line1,
        line2=line2,
        inclination_deg=float(line2[8:16]),
        raan_deg=float(line2[17:25]),
        eccentricity=float("0." + line2[26:33].strip()),
        arg_perigee_deg=float(line2[34:42]),
        mean_motion=float(line2[52:63]),
        bstar=sat.bstar,
        epoch=epoch_dt,
    )
