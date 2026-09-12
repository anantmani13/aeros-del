"""
Lagrangian Plume Transport Equations — WRF-Chem Derived

Implements:
- Single Lagrangian trajectory step (advection by wind)
- Full trajectory computation over time horizon
- Gaussian plume concentration dispersion
- Pasquill-Gifford dispersion coefficients (σy, σz)
"""

import numpy as np
from typing import List, Tuple, Optional


# ── Physical Constants ───────────────────────────────────────────────
EARTH_RADIUS_M = 6371000.0


def lagrangian_trajectory_step(
    lat: float,
    lon: float,
    u_wind: float,
    v_wind: float,
    dt_seconds: float = 3600.0,
) -> Tuple[float, float]:
    """
    Advance a Lagrangian particle by one time step under wind advection.

    Converts (u, v) winds to (dlat, dlon) displacement.

    Args:
        lat: Current latitude (degrees)
        lon: Current longitude (degrees)
        u_wind: Eastward wind component (m/s)
        v_wind: Northward wind component (m/s)
        dt_seconds: Time step (s), default 1 hour

    Returns:
        Tuple of (new_lat, new_lon) in degrees
    """
    # Displacement in meters
    dx = u_wind * dt_seconds  # eastward
    dy = v_wind * dt_seconds  # northward

    # Convert to degrees
    dlat = dy / EARTH_RADIUS_M * (180.0 / np.pi)
    dlon = dx / (EARTH_RADIUS_M * np.cos(np.radians(lat))) * (180.0 / np.pi)

    return lat + dlat, lon + dlon


def compute_full_trajectory(
    start_lat: float,
    start_lon: float,
    u_winds: List[float],
    v_winds: List[float],
    dt_seconds: float = 3600.0,
    max_steps: Optional[int] = None,
) -> List[dict]:
    """
    Compute full forward trajectory from a source point.

    Args:
        start_lat: Source latitude (degrees)
        start_lon: Source longitude (degrees)
        u_winds: List of u-wind values at each time step (m/s)
        v_winds: List of v-wind values at each time step (m/s)
        dt_seconds: Time step (s)
        max_steps: Maximum number of steps (defaults to len of wind data)

    Returns:
        List of trajectory points, each a dict with lat, lon, hour, distance_km
    """
    n_steps = min(len(u_winds), len(v_winds))
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    trajectory = [{
        "lat": start_lat,
        "lon": start_lon,
        "hour": 0,
        "distance_km": 0.0,
    }]

    lat, lon = start_lat, start_lon
    total_dist = 0.0

    for i in range(n_steps):
        # Calculate displacement
        dx = u_winds[i] * dt_seconds
        dy = v_winds[i] * dt_seconds
        step_dist = np.sqrt(dx ** 2 + dy ** 2) / 1000.0  # km
        total_dist += step_dist

        lat, lon = lagrangian_trajectory_step(
            lat, lon, u_winds[i], v_winds[i], dt_seconds
        )

        trajectory.append({
            "lat": float(lat),
            "lon": float(lon),
            "hour": (i + 1) * dt_seconds / 3600.0,
            "distance_km": round(total_dist, 2),
        })

    return trajectory


def pasquill_gifford_sigma(
    downwind_distance_m: float,
    stability_class: str = "D",
) -> Tuple[float, float]:
    """
    Calculate Pasquill-Gifford dispersion coefficients σy and σz.

    Based on stability class (A-F):
    - A: Very unstable (strong convection)
    - B: Moderately unstable
    - C: Slightly unstable
    - D: Neutral (default)
    - E: Slightly stable
    - F: Very stable (strong inversion — Delhi winter nights)

    Args:
        downwind_distance_m: Downwind distance from source (m)
        stability_class: Pasquill stability class (A-F)

    Returns:
        Tuple of (sigma_y, sigma_z) in meters
    """
    x_km = max(downwind_distance_m / 1000.0, 0.01)

    # Empirical coefficients: sigma_y = a * x^b, sigma_z = c * x^d
    pg_params = {
        "A": {"a": 209.6, "b": 0.9011, "c": 417.9, "d": 2.058},
        "B": {"a": 154.7, "b": 0.8942, "c": 109.3, "d": 1.064},
        "C": {"a": 103.3, "b": 0.9112, "c": 61.14, "d": 0.9147},
        "D": {"a": 68.28, "b": 0.9112, "c": 30.37, "d": 0.7306},
        "E": {"a": 51.05, "b": 0.9112, "c": 21.14, "d": 0.6780},
        "F": {"a": 33.96, "b": 0.9112, "c": 13.72, "d": 0.6210},
    }

    params = pg_params.get(stability_class.upper(), pg_params["D"])

    sigma_y = params["a"] * x_km ** params["b"]
    sigma_z = params["c"] * x_km ** params["d"]

    # Cap sigma_z to prevent unrealistic vertical spread
    sigma_z = min(sigma_z, 5000.0)

    return float(sigma_y), float(sigma_z)


def gaussian_plume_concentration(
    x: float,
    y: float,
    z: float,
    source_emission_rate: float,
    wind_speed: float,
    effective_stack_height: float = 0.0,
    stability_class: str = "D",
) -> float:
    """
    Calculate concentration at (x, y, z) from a Gaussian plume model.

    C(x,y,z) = (Q / (2π u σy σz)) * exp(-y²/(2σy²))
               * [exp(-(z-H)²/(2σz²)) + exp(-(z+H)²/(2σz²))]

    Includes ground reflection term.

    Args:
        x: Downwind distance from source (m)
        y: Crosswind distance from plume centerline (m)
        z: Height above ground for concentration (m)
        source_emission_rate: Source emission rate Q (µg/s)
        wind_speed: Wind speed at effective height (m/s)
        effective_stack_height: Effective source height H (m), 0 for ground-level
        stability_class: Pasquill stability class (A-F)

    Returns:
        Concentration (µg/m³) at the given point
    """
    if x <= 0 or wind_speed <= 0:
        return 0.0

    sigma_y, sigma_z = pasquill_gifford_sigma(x, stability_class)

    # Prevent division by zero
    sigma_y = max(sigma_y, 0.1)
    sigma_z = max(sigma_z, 0.1)
    wind_speed = max(wind_speed, 0.1)

    H = effective_stack_height

    # Gaussian plume equation
    coeff = source_emission_rate / (2.0 * np.pi * wind_speed * sigma_y * sigma_z)

    # Crosswind dispersion
    lateral = np.exp(-y ** 2 / (2.0 * sigma_y ** 2))

    # Vertical dispersion with ground reflection
    vertical = (
        np.exp(-(z - H) ** 2 / (2.0 * sigma_z ** 2))
        + np.exp(-(z + H) ** 2 / (2.0 * sigma_z ** 2))
    )

    concentration = coeff * lateral * vertical

    return max(float(concentration), 0.0)
