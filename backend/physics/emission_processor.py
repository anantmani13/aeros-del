"""
Emission Processor Service — Speciation, Diurnal Modulation & Fire Gridding

Processes baseline anthropogenic emissions for Delhi NCR and integrates
biomass-burning plume contributions. Produces:
- Sector-resolved emission estimates for the current hour
- Diurnal-modulated pollutant totals
- Fire emission contributions from FRP-derived rates
- Grid summary for dashboard display
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.formulas.emission_formulas import (
    diurnal_modulation,
    speciation_profile,
    fire_emission_rate,
    total_gridded_emission,
)

logger = logging.getLogger(__name__)

# Baseline annual-average emission for a Delhi NCR grid cell
# (µg/m³/h at a reference height) — order-of-magnitude sector split.
BASE_EMISSIONS = {
    "vehicular": 18.0,
    "industrial": 8.0,
    "domestic": 6.0,
    "construction": 5.0,
}

SEASON_FACTOR = {
    1: 1.6, 2: 1.4, 3: 1.2, 4: 1.0, 5: 1.0, 6: 1.0,
    7: 0.7, 8: 0.7, 9: 0.9, 10: 1.2, 11: 1.8, 12: 1.7,
}

GRID_SIZE = 7  # 7x7 grid over Delhi NCR (~5 km cells)


class EmissionProcessor:
    """
    Emission speciation & gridding service.

    Consumes the current hour and optional fire data to estimate the
    present emission mix and gridded cell totals for the domain.
    """

    def __init__(self):
        self.grid_size = GRID_SIZE

    async def compute(
        self,
        fires: Optional[List[Dict]] = None,
        hour_ist: Optional[int] = None,
    ) -> Dict:
        """
        Compute current emission snapshot.

        Args:
            fires: Fire hotspot list (for FRP-derived emissions).
            hour_ist: Local hour (0-23); defaults to now.

        Returns:
            Dict with sector breakdown, speciated pollutants,
            fire contribution, and gridded emission array summary.
        """
        fires = fires or []
        if hour_ist is None:
            hour_ist = datetime.now(timezone.utc).astimezone().hour

        month = datetime.now().month
        season = SEASON_FACTOR.get(month, 1.0)

        # 1. Anthropogenic sectors
        sectors = {}
        pollutant_totals = {"pm25": 0.0, "pm10": 0.0, "no2": 0.0,
                            "so2": 0.0, "co": 0.0, "bc": 0.0}

        for sector, base in BASE_EMISSIONS.items():
            emission = total_gridded_emission(
                base_emission=base,
                hour_ist=hour_ist,
                sector=sector,
                season_factor=season,
            )
            species = speciation_profile(sector, emission)
            sectors[sector] = {
                "emission_rate": round(emission, 2),
                "species": {k: round(v, 3) for k, v in species.items()},
            }
            for key in pollutant_totals:
                pollutant_totals[key] += species.get(key, 0.0)

        # 2. Biomass burning contribution
        summary = self._aggregate_fire_emissions(fires, hour_ist)
        pollutant_totals["pm25"] += summary["pm25"]
        pollutant_totals["pm10"] += summary["pm10"]
        pollutant_totals["co"] += summary["co"]
        pollutant_totals["bc"] += summary["bc"]

        grid = self._build_grid(pollutant_totals["pm25"])

        return {
            "hour_ist": hour_ist,
            "season_factor": season,
            "sectors": sectors,
            "pollutant_totals": {k: round(v, 3) for k, v in pollutant_totals.items()},
            "fire_contribution": {
                "active_fires": len(fires),
                "total_frp_mw": round(sum(
                    self._to_float(f.get("frp"), 0.0) for f in fires
                ), 1),
                "pm25_kg_hr": round(summary["pm25_kg_hr"], 1),
                "pm25_ug_m3": round(summary["pm25"], 3),
            },
            "grid": grid,
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _aggregate_fire_emissions(
        self, fires: List[Dict], hour_ist: int
    ) -> Dict:
        """Sum FRP-derived emission rates, modulated by burning hour."""
        total_pm25_kg = 0.0
        total_pm10_kg = 0.0
        total_co_kg = 0.0
        total_bc_kg = 0.0

        modulation = diurnal_modulation(hour_ist, "biomass_burning")

        for fire in fires:
            frp = self._to_float(fire.get("frp"), 0.0)
            if frp <= 0:
                continue
            rates = fire_emission_rate(frp)
            total_pm25_kg += rates["pm25_kg_hr"] * modulation
            total_pm10_kg += rates["pm10_kg_hr"] * modulation
            total_co_kg += rates["co_kg_hr"] * modulation
            total_bc_kg += rates["bc_kg_hr"] * modulation

        # Convert domain total (kg/hr) to a per-cell surface increment.
        # Assumes fire plume mass gets dispersed over the domain volume.
        domain_volume_m3 = 80_000 * 80_000 * 1000  # 80km x 80km x 1km
        pm25_ug_m3 = total_pm25_kg * 1e9 / max(domain_volume_m3, 1)

        return {
            "pm25": pm25_ug_m3,
            "pm10": total_pm10_kg * 1e9 / max(domain_volume_m3, 1),
            "co": total_co_kg * 1e9 / max(domain_volume_m3, 1),
            "bc": total_bc_kg * 1e9 / max(domain_volume_m3, 1),
            "pm25_kg_hr": total_pm25_kg,
        }

    def _build_grid(self, pm25_center: float) -> Dict:
        """Build a coarse concentration grid summary for the domain."""
        cells = []
        lat_min, lon_min = 28.30, 76.80
        dlat = (28.90 - 28.30) / self.grid_size
        dlon = (77.50 - 76.80) / self.grid_size

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                lat = lat_min + i * dlat
                lon = lon_min + j * dlon
                # Simple spatial decay from Delhi center
                center_km = self._haversine(28.6139, 77.2090, lat, lon)
                factor = max(0.35, 1.0 - center_km / 80.0)
                cells.append({
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "pm25_est": round(pm25_center * factor, 1),
                })

        return {
            "nx": self.grid_size,
            "ny": self.grid_size,
            "cell_count": len(cells),
            "cells": cells,
        }

    def _haversine(self, lat1, lon1, lat2, lon2) -> float:
        import math
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _to_float(self, value, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default