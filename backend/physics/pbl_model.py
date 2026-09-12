"""
PBL Model Service — Planetary Boundary Layer Height & Inversion Diagnostics

Combines surface meteorology with WRF-Chem-derived equations to estimate:
- Potential temperature and virtual potential temperature profiles
- Bulk Richardson number stability metrics
- PBL height (from ERA5 boundary_layer_height when available, else Richardson)
- Nocturnal inversion strength (ΔT / 100m)

Handles missing upper-air data by constructing a physically plausible
diurnal profile based on surface observations.
"""

import asyncio
import logging
from typing import Dict, List, Optional

import numpy as np

from backend.formulas.pbl_formulas import (
    potential_temperature,
    virtual_potential_temperature,
    bulk_richardson_number,
    estimate_pbl_height,
    inversion_strength,
)

logger = logging.getLogger(__name__)

# Default dry-adiabatic lapse rate (°C/100m) and typical winter Delhi
# behavior used to synthesize upper-air profiles when measurements lack
# the required levels.
DRY_LAPSE_C_PER_100M = 0.98
NOCTURNAL_INVERSION_C = 1.5   # typical ΔT across lowest 100m on calm nights


class PBLModel:
    """
    Planetary boundary layer height & inversion diagnostics service.

    Consumes a surface weather snapshot (optionally with an observed
    PBL height from ERA5/Open-Meteo) and produces a physics-informed
    PBL assessment for use by the AISI calculator and ML features.
    """

    def __init__(self, z0: float = 1.0, u_star: float = 0.3):
        self.z0 = z0
        self.u_star = u_star

    async def compute(
        self,
        weather: Dict,
        hour_ist: Optional[int] = None,
    ) -> Dict:
        """
        Compute PBL diagnostics from a weather snapshot.

        Args:
            weather: Dict with keys from WeatherClient, at minimum
                temperature_c, pressure_hpa, wind_speed_ms,
                wind_direction_deg, and optionally pbl_height_m.
            hour_ist: Local hour (0-23); defaults to current IST hour.

        Returns:
            Dict with pbl_height_m, inversion_strength_k, ri_bulk,
            stability_class, potential_temperature, and diagnostics.
        """
        try:
            t_sfc = self._to_float(weather.get("temperature_c"), default=25.0)
            p_sfc = self._to_float(weather.get("pressure_hpa"), default=1013.0)
            rh = self._to_float(weather.get("relative_humidity"), default=50.0)
            u = self._to_float(weather.get("u_wind_speed", weather.get("wind_speed_ms")), default=2.0)
            v = self._to_float(weather.get("v_wind_speed"), default=0.0)
            pbl_obs = self._to_float(weather.get("pbl_height_m"), default=None)

            if hour_ist is None:
                from datetime import datetime, timezone
                try:
                    from zoneinfo import ZoneInfo
                    hour_ist = datetime.now(timezone.utc).astimezone(
                        ZoneInfo("Asia/Kolkata")).hour
                except Exception:
                    # UTC+5:30 fallback when tzdata is unavailable
                    hour_ist = (datetime.now(timezone.utc).hour + 5) % 24 + 1
                    hour_ist %= 24

            theta_sfc = potential_temperature(t_sfc + 273.15, p_sfc * 100.0)

            # Synthesize a modest upper-air profile. During nighttime a
            # surface inversion is modelled; during daytime the profile
            # warms dry-adiabatically in the mixed layer.
            t_100m = self._estimate_temperature_at_100m(
                t_sfc, hour_ist, pbl_obs
            )
            inversion_k = inversion_strength(t_sfc + 273.15, t_100m + 273.15)

            # u,v components from weather if directly provided
            u_wind, v_wind = self._wind_components(weather, u, v)

            theta_v_sfc = virtual_potential_temperature(theta_sfc, self._mixing_ratio(rh, p_sfc, t_sfc))
            theta_v_top = virtual_potential_temperature(
                potential_temperature(t_100m + 273.15, (p_sfc - 1.0) * 100.0),
                self._mixing_ratio(rh, p_sfc - 1.0, t_100m),
            )

            ri_bulk = bulk_richardson_number(
                theta_v_surface=theta_v_sfc,
                theta_v_z=theta_v_top,
                z=100.0,
                z0=self.z0,
                u_z=u_wind,
                v_z=v_wind,
                u_star=self.u_star,
            )

            # Prefer observed ERA5 PBL height when sensible; otherwise
            # fall back to a stability-scaled estimate.
            pbl_height = self._finalize_pbl_height(pbl_obs, ri_bulk, inversion_k, hour_ist)

            stability = self._stability_class(inversion_k, ri_bulk, hour_ist)

            return {
                "pbl_height_m": round(pbl_height, 1),
                "inversion_strength_k": round(inversion_k, 3),
                "ri_bulk": round(ri_bulk, 4),
                "stability_class": stability,
                "potential_temperature_k": round(theta_sfc, 2),
                "t_100m_c": round(t_100m, 2),
                "source": "pbl_model",
            }

        except Exception as e:
            logger.error(f"PBL computation failed: {e}")
            return self._fallback_diagnostics()

    # ── Helpers ──────────────────────────────────────────────────────

    def _estimate_temperature_at_100m(
        self,
        t_sfc: float,
        hour_ist: int,
        pbl_obs: Optional[float],
    ) -> float:
        """Estimate temperature at 100 m AGL using a smooth diurnal model.

        A cosine blend replaces the old night/day step function, which
        injected ~1.5 K (∼3.8 AISI points) discontinuities at 07/11/17/20h.
        """
        import math
        # Night weight: 1 at 02h, 0 at 14h (smooth 24h cosine)
        night_w = 0.5 * (1.0 + math.cos((hour_ist - 2) * math.pi / 12.0))
        night_w = max(0.0, min(1.0, night_w))
        calm_factor = 1.0 if (pbl_obs is not None and pbl_obs < 400) else 0.8
        night_t = t_sfc + NOCTURNAL_INVERSION_C * calm_factor
        day_t = t_sfc - DRY_LAPSE_C_PER_100M
        return night_w * night_t + (1.0 - night_w) * day_t

    def _mixing_ratio(self, rh: float, pressure_hpa: float, t_c: float) -> float:
        """Approximate water-vapor mixing ratio (kg/kg) from RH."""
        es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))  # hPa
        qv = (0.622 * rh / 100.0 * es) / max(pressure_hpa - 0.378 * rh / 100.0 * es, 1.0)
        return max(float(qv), 0.0)

    def _wind_components(self, weather: Dict, u: float, v: float) -> tuple:
        """Extract u/v wind from either component keys or speed+direction."""
        import math

        if "u_wind" in weather and "v_wind" in weather:
            return (
                self._to_float(weather.get("u_wind"), u),
                self._to_float(weather.get("v_wind"), v),
            )
        spd = self._to_float(weather.get("wind_speed_ms"), u)
        direction = self._to_float(weather.get("wind_direction_deg"), 270.0)
        rad = math.radians(direction)
        return -spd * math.sin(rad), -spd * math.cos(rad)

    def _finalize_pbl_height(
        self,
        pbl_obs: Optional[float],
        ri_bulk: float,
        inversion_k: float,
        hour_ist: int,
    ) -> float:
        """Return a sanitized PBL height estimate."""
        if pbl_obs is not None and 50.0 <= pbl_obs <= 3000.0:
            # Blend observed value with stability indicator
            if inversion_k > 0.5 and pbl_obs > 800:
                # Inversion reported but deep PBL is inconsistent — cap it
                return max(pbl_obs * 0.6, 200.0)
            return pbl_obs

        # Heuristic fallback (no observed PBL): smooth cosine between a
        # 350 m nocturnal minimum and a 1200 m convective maximum.
        import math
        day_w = 0.5 * (1.0 - math.cos((hour_ist - 2) * math.pi / 12.0))
        day_w = max(0.0, min(1.0, day_w))
        base = 350.0 + (1200.0 - 350.0) * day_w
        if inversion_k > 0.5:
            base *= 0.5
        if ri_bulk > 0.25:
            base *= 0.6
        return max(base, 50.0)

    def _stability_class(self, inversion_k: float, ri_bulk: float, hour_ist: int) -> str:
        """Map diagnostics to Pasquill-style stability class."""
        night = hour_ist in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
        if night and (inversion_k > 1.0 or ri_bulk > 0.25):
            return "F"  # Very stable
        if night:
            return "E"
        if ri_bulk > 0.10:
            return "D"  # Neutral
        if 11 <= hour_ist <= 16:
            return "B"  # Moderately unstable
        return "C"

    def _to_float(self, value, default: float) -> float:
        """Coerce a value to float or return default."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _fallback_diagnostics(self) -> Dict:
        """Return safe default diagnostics on failure."""
        return {
            "pbl_height_m": 700.0,
            "inversion_strength_k": 0.5,
            "ri_bulk": 0.1,
            "stability_class": "D",
            "potential_temperature_k": 300.0,
            "t_100m_c": 25.5,
            "source": "fallback",
        }