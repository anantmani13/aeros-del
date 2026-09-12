"""
Aerosol-Radiation Interaction Equations — WRF-Chem Derived

Implements:
- AOD estimation from surface PM2.5
- Surface radiative forcing from aerosol loading
- Black carbon absorption warming
- PBL height suppression feedback (PM↑ → dimming → PBL collapse → PM↑↑)
"""

import numpy as np
from typing import Tuple


# ── Physical Constants ───────────────────────────────────────────────
S0 = 1361.0       # Solar constant (W/m²)
RHO_AIR = 1.225   # Air density at sea level (kg/m³)


def aod_from_pm25(
    pm25: float,
    effective_height_m: float = 1000.0,
    extinction_coeff: float = 3.5e-6,
    rh_factor: float = 1.0,
) -> float:
    """
    Estimate Aerosol Optical Depth (AOD) from surface PM2.5.

    AOD = σ_ext * PM2.5 * H_eff / ρ_air

    Empirical relationship calibrated for Delhi NCR using
    MODIS/AERONET AOD vs CPCB PM2.5 correlations.

    Args:
        pm25: Surface PM2.5 concentration (µg/m³)
        effective_height_m: Effective aerosol layer height (m)
        extinction_coeff: Mass extinction coefficient (m²/µg)
        rh_factor: Relative humidity hygroscopic growth factor (1.0-2.5)

    Returns:
        Estimated column AOD at 550nm (dimensionless)
    """
    # Convert µg/m³ to kg/m³ for dimensional consistency
    pm25_kgm3 = pm25 * 1e-9

    aod = extinction_coeff * rh_factor * pm25 * effective_height_m / RHO_AIR

    # Empirical bounds (MODIS validation for Delhi: AOD rarely > 5)
    return max(0.0, min(aod, 5.0))


def surface_radiative_forcing(
    aod: float,
    solar_zenith_deg: float,
    surface_albedo: float = 0.15,
    single_scatter_albedo: float = 0.88,
) -> float:
    """
    Calculate surface shortwave radiative forcing due to aerosols.

    ΔF = -α_s * S0 * cos(θ) * (1 - exp(-τ * sec(θ)))

    Negative = cooling at surface (dimming effect).

    Args:
        aod: Aerosol optical depth (dimensionless)
        solar_zenith_deg: Solar zenith angle (degrees)
        surface_albedo: Surface albedo (0-1)
        single_scatter_albedo: Aerosol single scattering albedo (0-1)

    Returns:
        Surface radiative forcing (W/m²), negative = dimming
    """
    theta_rad = np.radians(solar_zenith_deg)
    cos_theta = max(np.cos(theta_rad), 0.01)  # avoid night issues
    sec_theta = 1.0 / cos_theta

    # Simplified two-stream approximation
    forcing = -(1.0 - surface_albedo) * S0 * cos_theta * (
        1.0 - np.exp(-aod * sec_theta)
    )

    # Scale by single scatter albedo (more absorbing = less scattered back)
    forcing *= single_scatter_albedo

    # Typical range for Delhi: -50 to -120 W/m² during heavy haze
    return float(forcing)


def bc_absorption_warming(
    pm25: float,
    bc_fraction: float = 0.08,
    absorption_efficiency: float = 7.5,
    layer_depth_m: float = 500.0,
) -> float:
    """
    Calculate atmospheric heating rate from black carbon absorption.

    BC heating rate ~= (Q_abs * BC_mass * S) / (ρ_air * Cp * Δz)

    Delhi NCR has high BC fraction from vehicular & biomass sources.

    Args:
        pm25: PM2.5 concentration (µg/m³)
        bc_fraction: Black carbon fraction of PM2.5 (typical 0.05-0.15)
        absorption_efficiency: BC mass absorption cross-section (m²/g)
        layer_depth_m: Depth of absorbing layer (m)

    Returns:
        Heating rate (°C/day)
    """
    bc_conc = pm25 * bc_fraction  # µg/m³

    # Absorption coefficient (1/m)
    abs_coeff = bc_conc * absorption_efficiency * 1e-6

    # Heating rate (simplified from radiative transfer)
    # H = (S0 * abs_coeff) / (ρ_air * Cp)  [K/s]  → convert to °C/day
    heating_rate_ks = (S0 * 0.5 * abs_coeff) / (RHO_AIR * 1004.0)
    heating_rate_day = heating_rate_ks * 86400.0

    # Typical range for Delhi winter: +1 to +2.5 °C/day
    return max(0.0, min(heating_rate_day, 5.0))


def pbl_suppression_factor(
    aod: float,
    baseline_pbl_m: float = 1500.0,
) -> Tuple[float, float]:
    """
    Calculate PBL height suppression due to aerosol-radiation feedback.

    The positive feedback loop:
    PM↑ → surface dimming↑ → reduced surface heating → PBL collapse → PM↑↑

    Based on studies showing 20-40% PBL reduction during high-AOD events in IGP.

    Args:
        aod: Aerosol optical depth
        baseline_pbl_m: Clear-sky PBL height (m)

    Returns:
        Tuple of (suppressed_pbl_height_m, suppression_fraction)
    """
    # Suppression fraction increases with AOD (empirical sigmoid)
    # At AOD=1.0 → ~25% suppression; AOD=3.0 → ~45% suppression
    suppression = 0.5 * (1.0 - np.exp(-0.6 * aod))

    suppressed_pbl = baseline_pbl_m * (1.0 - suppression)
    suppressed_pbl = max(suppressed_pbl, 50.0)  # minimum PBL height

    return float(suppressed_pbl), float(suppression)
