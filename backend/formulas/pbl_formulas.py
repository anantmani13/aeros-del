"""
PBL & Richardson Number Equations — WRF-Chem Derived

Implements boundary layer diagnostics:
- Potential temperature (θ) from pressure & temperature
- Virtual potential temperature (θ_v) accounting for moisture
- Bulk Richardson Number (Ri_b) for stability classification
- PBL height estimation from Ri_b profile
- Inversion strength (ΔT/100m)
"""

import numpy as np
from typing import Tuple, Optional

# ── Physical Constants ───────────────────────────────────────────────
G = 9.81          # gravitational acceleration (m/s²)
P0 = 100000.0     # reference pressure (Pa)
RD = 287.05       # gas constant for dry air (J/kg/K)
CP = 1004.0       # specific heat at constant pressure (J/kg/K)
KAPPA = RD / CP   # Poisson constant (~0.286)
RI_CRIT = 0.25    # critical Richardson number for PBL top
ETA = 100.0       # surface friction coefficient for Ri_b


def potential_temperature(temperature_k: float, pressure_pa: float) -> float:
    """
    Calculate potential temperature θ.

    θ = T * (P₀ / P)^κ

    Args:
        temperature_k: Air temperature (K)
        pressure_pa: Atmospheric pressure (Pa)

    Returns:
        Potential temperature (K)
    """
    return temperature_k * (P0 / pressure_pa) ** KAPPA


def virtual_potential_temperature(
    theta: float,
    mixing_ratio: float = 0.0
) -> float:
    """
    Calculate virtual potential temperature θ_v.

    θ_v = θ * (1 + 0.61 * r)

    where r is the water vapor mixing ratio (kg/kg).

    Args:
        theta: Potential temperature (K)
        mixing_ratio: Water vapor mixing ratio (kg/kg), default 0 (dry)

    Returns:
        Virtual potential temperature (K)
    """
    return theta * (1.0 + 0.61 * mixing_ratio)


def bulk_richardson_number(
    theta_v_surface: float,
    theta_v_z: float,
    z: float,
    z0: float,
    u_z: float,
    v_z: float,
    u_0: float = 0.0,
    v_0: float = 0.0,
    u_star: float = 0.3,
) -> float:
    """
    Calculate Bulk Richardson Number.

    Ri_b(z) = (g / θ_v0) * (θ_vz - θ_v0) * (z - z0)
              / [(u_z - u_0)² + (v_z - v_0)² + η * u*²]

    Args:
        theta_v_surface: Virtual potential temperature at surface (K)
        theta_v_z: Virtual potential temperature at height z (K)
        z: Height above ground (m)
        z0: Surface roughness length (m)
        u_z: U-component of wind at height z (m/s)
        v_z: V-component of wind at height z (m/s)
        u_0: U-component of wind at surface (m/s)
        v_0: V-component of wind at surface (m/s)
        u_star: Friction velocity (m/s)

    Returns:
        Bulk Richardson number (dimensionless)
    """
    dz = max(z - z0, 1.0)
    d_theta_v = theta_v_z - theta_v_surface
    shear_sq = (u_z - u_0) ** 2 + (v_z - v_0) ** 2 + ETA * u_star ** 2

    if shear_sq < 1e-6:
        shear_sq = 1e-6  # prevent division by zero

    ri_b = (G / theta_v_surface) * d_theta_v * dz / shear_sq
    return ri_b


def estimate_pbl_height(
    heights: np.ndarray,
    theta_v_profile: np.ndarray,
    u_profile: np.ndarray,
    v_profile: np.ndarray,
    z0: float = 1.0,
    u_star: float = 0.3,
) -> float:
    """
    Estimate PBL height as the level where Ri_b first exceeds Ri_crit (0.25).

    Uses linear interpolation between levels bracketing the critical threshold.

    Args:
        heights: Array of heights above ground (m), ascending
        theta_v_profile: Array of virtual potential temperatures at each height (K)
        u_profile: Array of u-wind components at each height (m/s)
        v_profile: Array of v-wind components at each height (m/s)
        z0: Surface roughness length (m)
        u_star: Friction velocity (m/s)

    Returns:
        Estimated PBL height (m). Returns max height if Ri_crit never exceeded.
    """
    theta_v0 = theta_v_profile[0]

    for i in range(1, len(heights)):
        ri_b = bulk_richardson_number(
            theta_v_surface=theta_v0,
            theta_v_z=theta_v_profile[i],
            z=heights[i],
            z0=z0,
            u_z=u_profile[i],
            v_z=v_profile[i],
            u_star=u_star,
        )

        if ri_b >= RI_CRIT:
            # Linear interpolation to find exact crossing
            ri_b_prev = bulk_richardson_number(
                theta_v_surface=theta_v0,
                theta_v_z=theta_v_profile[i - 1],
                z=heights[i - 1],
                z0=z0,
                u_z=u_profile[i - 1],
                v_z=v_profile[i - 1],
                u_star=u_star,
            )
            # Interpolate between levels i-1 and i
            frac = (RI_CRIT - ri_b_prev) / max(ri_b - ri_b_prev, 1e-10)
            pbl_h = heights[i - 1] + frac * (heights[i] - heights[i - 1])
            return max(pbl_h, 50.0)  # minimum 50m

    return float(heights[-1])


def inversion_strength(
    t_surface_k: float,
    t_100m_k: float,
) -> float:
    """
    Calculate temperature inversion strength ΔT/100m.

    Positive values indicate inversion (temperature increases with height).

    Args:
        t_surface_k: Temperature at surface (K)
        t_100m_k: Temperature at 100m AGL (K)

    Returns:
        Inversion strength (K/100m). Positive = inversion present.
    """
    return t_100m_k - t_surface_k
