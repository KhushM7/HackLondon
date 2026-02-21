"""Verify that all required dependencies are importable and functional."""
import sys


def verify() -> bool:
    errors = []

    try:
        import numpy as np
        _ = np.array([1.0, 2.0, 3.0])
        print(f"  numpy {np.__version__} OK")
    except Exception as e:
        errors.append(f"numpy: {e}")

    try:
        import scipy
        from scipy.optimize import brentq  # noqa: F401
        from scipy.integrate import solve_ivp, dblquad  # noqa: F401
        from scipy.stats import norm  # noqa: F401
        print(f"  scipy {scipy.__version__} OK")
    except Exception as e:
        errors.append(f"scipy: {e}")

    try:
        import sgp4
        from sgp4.api import Satrec, jday  # noqa: F401
        print(f"  sgp4 {sgp4.__version__} OK")
    except Exception as e:
        errors.append(f"sgp4: {e}")

    try:
        import astropy
        from astropy import units as u  # noqa: F401
        from astropy.time import Time  # noqa: F401
        from astropy.coordinates import TEME, GCRS, ITRS, CartesianRepresentation  # noqa: F401
        print(f"  astropy {astropy.__version__} OK")
    except Exception as e:
        errors.append(f"astropy: {e}")

    try:
        from sqlalchemy import __version__ as sa_ver
        print(f"  sqlalchemy {sa_ver} OK")
    except Exception as e:
        errors.append(f"sqlalchemy: {e}")

    try:
        from fastapi import __version__ as fa_ver
        print(f"  fastapi {fa_ver} OK")
    except Exception as e:
        errors.append(f"fastapi: {e}")

    if errors:
        print("\nFAILED imports:")
        for err in errors:
            print(f"  ✗ {err}")
        return False

    print("\nAll dependencies verified successfully.")
    return True


if __name__ == "__main__":
    print("Verifying OrbitGuard dependencies...\n")
    ok = verify()
    sys.exit(0 if ok else 1)
