"""
Emission Processing Equations — WRF-Chem Derived

Implements:
- Diurnal modulation profiles for different emission sectors
- Speciation profiles (vehicular, industrial, domestic, construction)
- Fire emission rate estimation from FRP
- Total gridded emission computation
"""

import numpy as np
from typing import Dict, Optional


# ──────────────────────────────────────────────────────────────────────
# Diurnal Modulation Profiles
# ──────────────────────────────────────────────────────────────────────
# Hourly scaling factors (0-23h IST) for Delhi NCR emission sectors.
# Sum of 24 values = 24.0 (average factor = 1.0)

DIURNAL_PROFILES = {
    "traffic": [
        0.3, 0.2, 0.15, 0.15, 0.2, 0.5,   # 00-05: low nighttime
        1.0, 1.8, 2.2, 1.8, 1.2, 1.0,      # 06-11: morning rush
        0.9, 0.8, 0.8, 0.9, 1.2, 1.8,      # 12-17: afternoon build
        2.2, 2.0, 1.5, 1.0, 0.6, 0.4,      # 18-23: evening rush decay
    ],
    "industrial": [
        0.4, 0.4, 0.4, 0.4, 0.5, 0.6,      # 00-05: minimal
        0.8, 1.0, 1.2, 1.4, 1.5, 1.5,      # 06-11: ramp-up
        1.5, 1.5, 1.4, 1.3, 1.2, 1.0,      # 12-17: peak operations
        0.8, 0.6, 0.5, 0.4, 0.4, 0.4,      # 18-23: wind-down
    ],
    "domestic": [
        0.2, 0.1, 0.1, 0.1, 0.2, 0.5,      # 00-05: minimal
        1.5, 2.2, 1.8, 0.8, 0.5, 0.8,      # 06-11: morning cooking
        1.5, 0.8, 0.5, 0.5, 0.8, 1.5,      # 12-17: afternoon
        2.5, 2.2, 1.5, 0.8, 0.5, 0.3,      # 18-23: evening cooking peak
    ],
    "construction": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # 00-05: no activity
        0.5, 1.0, 1.5, 2.0, 2.0, 2.0,      # 06-11: ramp-up
        2.0, 2.0, 2.0, 2.0, 1.5, 1.0,      # 12-17: peak construction
        0.5, 0.0, 0.0, 0.0, 0.0, 0.0,      # 18-23: stops at dusk
    ],
    "biomass_burning": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # 00-05: no burning
        0.2, 0.5, 1.0, 2.0, 3.0, 3.5,      # 06-11: morning fires
        3.0, 2.5, 2.0, 1.5, 1.0, 0.5,      # 12-17: afternoon decay
        0.2, 0.1, 0.0, 0.0, 0.0, 0.0,      # 18-23: stops
    ],
}


def diurnal_modulation(
    hour_ist: int,
    sector: str = "traffic",
) -> float:
    """
    Get diurnal modulation factor for a given hour and emission sector.

    Args:
        hour_ist: Hour of day in IST (0-23)
        sector: Emission sector ("traffic", "industrial", "domestic",
                "construction", "biomass_burning")

    Returns:
        Modulation factor (0-3.5, average ~1.0)
    """
    hour_ist = int(hour_ist) % 24
    profile = DIURNAL_PROFILES.get(sector, DIURNAL_PROFILES["traffic"])
    return profile[hour_ist]


# ──────────────────────────────────────────────────────────────────────
# Speciation Profiles
# ──────────────────────────────────────────────────────────────────────
# Fractional contribution of each pollutant from different sectors
# PM2.5 fraction of total PM, and pollutant ratios

SPECIATION_PROFILES = {
    "vehicular": {
        "pm25_fraction": 0.85,   # 85% of vehicular PM is PM2.5
        "pm10_fraction": 1.0,
        "no2_ratio": 0.70,      # High NOx from diesel
        "so2_ratio": 0.05,
        "co_ratio": 0.80,
        "bc_fraction": 0.15,    # High BC from diesel exhaust
    },
    "industrial": {
        "pm25_fraction": 0.60,
        "pm10_fraction": 1.0,
        "no2_ratio": 0.40,
        "so2_ratio": 0.60,      # High SO2 from coal/fuel
        "co_ratio": 0.30,
        "bc_fraction": 0.05,
    },
    "domestic": {
        "pm25_fraction": 0.90,  # Cooking smoke mostly fine
        "pm10_fraction": 1.0,
        "no2_ratio": 0.10,
        "so2_ratio": 0.05,
        "co_ratio": 0.90,      # Incomplete combustion
        "bc_fraction": 0.10,
    },
    "construction": {
        "pm25_fraction": 0.30,  # Mostly coarse dust
        "pm10_fraction": 1.0,
        "no2_ratio": 0.05,
        "so2_ratio": 0.02,
        "co_ratio": 0.05,
        "bc_fraction": 0.01,
    },
    "biomass_burning": {
        "pm25_fraction": 0.80,
        "pm10_fraction": 1.0,
        "no2_ratio": 0.15,
        "so2_ratio": 0.05,
        "co_ratio": 0.85,
        "bc_fraction": 0.12,
    },
}


def speciation_profile(
    sector: str,
    total_pm10_emission: float,
) -> Dict[str, float]:
    """
    Apply speciation profile to convert total PM10 emission into individual pollutants.

    Args:
        sector: Emission sector name
        total_pm10_emission: Total PM10 emission rate (µg/m³/h or tons/day)

    Returns:
        Dict with emission rates for each pollutant species
    """
    profile = SPECIATION_PROFILES.get(sector, SPECIATION_PROFILES["vehicular"])

    return {
        "pm25": total_pm10_emission * profile["pm25_fraction"],
        "pm10": total_pm10_emission * profile["pm10_fraction"],
        "no2": total_pm10_emission * profile["no2_ratio"],
        "so2": total_pm10_emission * profile["so2_ratio"],
        "co": total_pm10_emission * profile["co_ratio"] * 10,  # CO in mg scale
        "bc": total_pm10_emission * profile["bc_fraction"],
    }


def fire_emission_rate(
    frp_mw: float,
    emission_factor_kg_per_mj: float = 0.013,
    pm25_ef: float = 10.5,
) -> Dict[str, float]:
    """
    Estimate fire emission rate from Fire Radiative Power (FRP).

    Based on FINN v2.5 methodology:
    Emission = FRP * emission_factor * species_ef

    Args:
        frp_mw: Fire Radiative Power (MW)
        emission_factor_kg_per_mj: Biomass combustion rate per MJ (kg/MJ)
        pm25_ef: PM2.5 emission factor (g/kg biomass)

    Returns:
        Dict with emission rates (kg/hr) for key species
    """
    # Biomass burning rate (kg/hr) from FRP
    # FRP (MW) → energy rate (MJ/hr) → biomass rate
    biomass_rate = frp_mw * 3600.0 * emission_factor_kg_per_mj  # kg/hr

    return {
        "pm25_kg_hr": biomass_rate * pm25_ef / 1000.0,
        "pm10_kg_hr": biomass_rate * 12.0 / 1000.0,
        "co_kg_hr": biomass_rate * 80.0 / 1000.0,
        "no2_kg_hr": biomass_rate * 2.5 / 1000.0,
        "so2_kg_hr": biomass_rate * 0.5 / 1000.0,
        "bc_kg_hr": biomass_rate * 0.5 / 1000.0,
        "oc_kg_hr": biomass_rate * 5.0 / 1000.0,
        "biomass_burned_kg_hr": biomass_rate,
    }


def total_gridded_emission(
    base_emission: float,
    hour_ist: int,
    sector: str,
    season_factor: float = 1.0,
    fire_contribution: float = 0.0,
) -> float:
    """
    Calculate total gridded emission for a cell at a given hour.

    Total = base * diurnal(hour) * season_factor + fire_contribution

    Args:
        base_emission: Annual-average baseline emission (µg/m³/h)
        hour_ist: Hour of day IST (0-23)
        sector: Emission sector
        season_factor: Seasonal scaling (e.g., 1.5 for winter)
        fire_contribution: Additional fire-related emission (µg/m³/h)

    Returns:
        Total emission rate (µg/m³/h)
    """
    modulation = diurnal_modulation(hour_ist, sector)
    return base_emission * modulation * season_factor + fire_contribution
