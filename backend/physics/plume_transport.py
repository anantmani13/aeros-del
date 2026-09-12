"""
Plume Transport Model — Lagrangian Smoke Trajectories from Source to Delhi

Computes forward trajectories from biomass-burning fire hotspots
(primarily Punjab/Haryana) to Delhi NCR using an hourly wind field.
Produces:
- GeoJSON trajectory lines for the dashboard
- Estimated arrival time at Delhi
- Concentration contribution at Delhi (Gaussian plume dispersion)
- Transport corridor attribution for NLP alerts
"""

import asyncio
import logging
import math
from typing import Dict, List, Optional

from backend.formulas.plume_formulas import (
    lagrangian_trajectory_step,
    compute_full_trajectory,
    gaussian_plume_concentration,
)

logger = logging.getLogger(__name__)

DELHI_CENTER = {"lat": 28.6139, "lon": 77.2090}
MAX_TRACKED_FIRES = 12
TRANSPORT_WINDOW_HOURS = 24


class PlumeTransportModel:
    """
    Lagrangian plume trajectory model.

    Uses the fire client output plus an hourly wind forecast to trace
    smoke parcels from fire sources toward Delhi and estimate their
    contribution to surface PM2.5.
    """

    async def compute_trajectories(
        self,
        fires: List[Dict],
        wind_forecast: Optional[Dict] = None,
        max_hours: int = TRANSPORT_WINDOW_HOURS,
    ) -> Dict:
        """
        Compute forward trajectories from the most significant fires.

        Args:
            fires: List of fire hotspot dicts (FireClient output).
            wind_forecast: Hourly wind dict with u_wind / v_wind arrays,
                or {wind_speed_ms, wind_direction_deg} for a steady field.
            max_hours: Maximum trajectory duration in hours.

        Returns:
            Dict with trajectories (GeoJSON), stats, arrival estimates,
            and the strongest transport corridor.
        """
        if not fires:
            return self._empty_result()

        u_winds, v_winds = self._build_wind_series(wind_forecast, max_hours)

        # Rank fires by FRP and cap count
        ranked = sorted(
            fires, key=lambda f: self._to_float(f.get("frp"), 0.0), reverse=True
        )[:MAX_TRACKED_FIRES]

        features = []
        arrival_estimates = []
        total_frp = 0.0
        contributing_fires = 0

        for fire in ranked:
            frp = self._to_float(fire.get("frp"), 0.0)
            total_frp += frp
            lat = self._to_float(fire.get("latitude"), 30.5)
            lon = self._to_float(fire.get("longitude"), 75.5)

            trajectory = self._trace_to_delhi(lat, lon, u_winds, v_winds)

            if not trajectory:
                continue

            arrival, km_at_delhi, max_conc = self._estimate_delhi_impact(
                lat, lon, frp, u_winds, v_winds
            )

            if arrival is not None:
                arrival_estimates.append({
                    "fire_index": len(features),
                    "arrival_hours": arrival,
                    "distance_km": round(km_at_delhi, 1),
                    "estimated_contribution_pm25": round(max_conc, 2),
                })
                if max_conc > 1.0:
                    contributing_fires += 1

            features.append(self._build_feature(fire, trajectory, arrival,
                                                max_conc))

        corridor = self._dominant_corridor(u_winds, v_winds)

        return {
            "features": features,
            "geojson": {
                "type": "FeatureCollection",
                "features": features,
            },
            "stats": {
                "total_fires_tracked": len(features),
                "total_frp_mw": round(total_frp, 1),
                "fires_arriving_at_delhi": len(arrival_estimates),
                "contributing_fires": contributing_fires,
            },
            "arrival_estimates": arrival_estimates,
            "corridor": corridor,
        }

    # ── Internals ────────────────────────────────────────────────────

    def _build_wind_series(
        self,
        forecast: Optional[Dict],
        max_hours: int,
    ) -> tuple:
        """Derive hourly u/v wind series from forecast or defaults."""
        u_winds, v_winds = [], []

        if forecast:
            if forecast.get("u_wind") and forecast.get("v_wind"):
                u_winds = [self._to_float(x, 0.0) for x in forecast["u_wind"]][:max_hours]
                v_winds = [self._to_float(x, 0.0) for x in forecast["v_wind"]][:max_hours]
            elif forecast.get("wind_speed_ms") and forecast.get("wind_direction_deg"):
                sp = forecast["wind_speed_ms"]
                dr = forecast["wind_direction_deg"]
                if isinstance(sp, list):
                    for s, d in zip(sp, dr):
                        u_winds.append(self._wind_component(s, d)[0])
                        v_winds.append(self._wind_component(s, d)[1])
                else:
                    u, v = self._wind_component(sp, dr)
                    u_winds = [u] * max_hours
                    v_winds = [v] * max_hours

        # Steady north-westerly fallback (typical winter IGP transport)
        if not u_winds:
            u_wind, v_wind = self._wind_component(3.2, 290.0)
            u_winds = [u_wind] * max_hours
            v_winds = [v_wind] * max_hours

        if len(u_winds) < max_hours:
            u_winds += [u_winds[-1]] * (max_hours - len(u_winds))
        if len(v_winds) < max_hours:
            v_winds += [v_winds[-1]] * (max_hours - len(v_winds))

        return u_winds[:max_hours], v_winds[:max_hours]

    def _trace_to_delhi(
        self,
        lat: float,
        lon: float,
        u_winds: List[float],
        v_winds: List[float],
    ) -> List[Dict]:
        """Compute trajectory points and detect Delhi proximity."""
        points = compute_full_trajectory(lat, lon, u_winds, v_winds)
        delhi_lat, delhi_lon = DELHI_CENTER["lat"], DELHI_CENTER["lon"]

        for p in points:
            km = self._haversine(p["lat"], p["lon"], delhi_lat, delhi_lon)
            p["distance_to_delhi_km"] = round(km, 1)
            p["reached_delhi"] = km <= 60.0

        return points

    def _estimate_delhi_impact(
        self,
        lat: float,
        lon: float,
        frp: float,
        u_winds: List[float],
        v_winds: List[float],
        hrs: int = TRANSPORT_WINDOW_HOURS,
    ) -> tuple:
        """
        Estimate arrival hour, source→Delhi distance and worst-case
        Gaussian plume contribution at Delhi.
        """
        delhi_lat, delhi_lon = DELHI_CENTER["lat"], DELHI_CENTER["lon"]

        for h in range(1, min(hrs, len(u_winds)) + 1):
            end = self._advect_step(lat, lon, u_winds, v_winds, h)
            km = self._haversine(end[0], end[1], delhi_lat, delhi_lon)
            if km <= 60.0:
                # Emission rate proxy: FRP → PM2.5 g/s (empirical factor)
                q = max(frp * 12.0, 100.0)  # g/s
                spd = math.hypot(u_winds[h - 1], v_winds[h - 1])
                conc = gaussian_plume_concentration(
                    x=max(km * 1000.0, 100.0),
                    y=0.0,
                    z=2.0,
                    source_emission_rate=q,
                    wind_speed=spd,
                    effective_stack_height=50.0,
                    stability_class="D",
                )
                return h, km, conc

        return None, 0.0, 0.0

    def _advect_step(
        self,
        lat: float,
        lon: float,
        u_winds: List[float],
        v_winds: List[float],
        steps: int,
    ) -> tuple:
        """Advance a parcel by `steps` hourly wind steps."""
        for i in range(steps):
            lat, lon = lagrangian_trajectory_step(
                lat, lon, u_winds[i], v_winds[i], 3600.0
            )
        return lat, lon

    def _build_feature(
        self,
        fire: Dict,
        trajectory: List[Dict],
        arrival: Optional[int],
        conc: float,
    ) -> Dict:
        """Construct a GeoJSON line feature for a fire trajectory."""
        coords = [[p["lon"], p["lat"]] for p in trajectory]
        ts = fire.get("acq_date", "") + (f" {fire.get('acq_time', '')}" if fire.get("acq_time") else "")

        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "lat": fire.get("latitude"),
                "lon": fire.get("longitude"),
                "frp_mw": fire.get("frp", 0),
                "region": fire.get("region", "Other"),
                "confidence": fire.get("confidence", "nominal"),
                "acquired": ts,
                "arrival_hours": arrival,
                "estimated_contribution_pm25": round(conc, 2),
                "reached_delhi": any(p.get("reached_delhi") for p in trajectory),
                "component": "plume-trajectory",
            },
        }

    def _dominant_corridor(self, u_winds: List[float], v_winds: List[float]) -> Dict:
        """Identify the dominant transport direction approaching Delhi."""
        avg_u = sum(u_winds[:12]) / max(len(u_winds[:12]), 1)
        avg_v = sum(v_winds[:12]) / max(len(v_winds[:12]), 1)
        bearing = (math.degrees(math.atan2(-avg_u, -avg_v)) + 360.0) % 360.0

        if 315 <= bearing or bearing < 45:
            name = "North-Westerly (Punjab corridor)"
        elif 45 <= bearing < 135:
            name = "North-Easterly (UP corridor)"
        elif 135 <= bearing < 225:
            name = "South-Easterly (Haryana corridor)"
        else:
            name = "South-Westerly (Rajasthan corridor)"

        return {
            "bearing_deg": round(bearing, 1),
            "name": name,
            "mean_wind_speed_ms": round(math.hypot(avg_u, avg_v), 2),
        }

    def _wind_component(self, speed: float, direction: float) -> tuple:
        """Convert meteorological wind speed+direction to u/v components."""
        rad = math.radians(direction)
        return -speed * math.sin(rad), -speed * math.cos(rad)

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Great-circle distance in km."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _to_float(self, value, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _empty_result(self) -> Dict:
        """Return an empty (valid) result when no fires exist."""
        return {
            "features": [],
            "geojson": {"type": "FeatureCollection", "features": []},
            "stats": {
                "total_fires_tracked": 0,
                "total_frp_mw": 0,
                "fires_arriving_at_delhi": 0,
                "contributing_fires": 0,
            },
            "arrival_estimates": [],
            "corridor": {"bearing_deg": 0.0, "name": "Undetermined", "mean_wind_speed_ms": 0.0},
        }