"""
Feature Engineering Pipeline — Station-Hour Feature Vectors

Builds the ~250-500 dimensional feature vector per station-hour used by
the XGBoost and Temporal Fusion Transformer forecasters.

Feature groups:
- Pollutant history (last 24-72h PM2.5, PM10, NO2, O3)
- Lag statistics (24h/48h/72h rolling mean/max/min)
- Temporal (hour, day-of-week, month, season, holiday flag)
- Meteorology (temp, RH, wind, pressure)
- Fire / biomass (upwind count, total FRP, nearest distance)
- Physics-derived (AISI, PBL height, inversion strength, AOD)
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

POLLUTANTS = ["pm25", "pm10", "no2", "o3", "co"]

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Autumn",
}

SEASON_ENCODE = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Autumn": 3}


class FeatureEngineer:
    """
    Constructs flat feature vectors and (optionally) 2D sequence windows
    for LSTM/TFT models from aligned station data.
    """

    def __init__(self, history_window: int = 72, horizon: int = 72):
        self.history_window = history_window
        self.horizon = horizon

    def build_features(
        self,
        station_id: str,
        readings: List[Dict],
        weather: Optional[Dict] = None,
        fire_summary: Optional[Dict] = None,
        physics: Optional[Dict] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Build a single feature vector for the most recent station hour.

        Args:
            station_id: Station identifier.
            readings: Chronological list of reading dicts (each with
                `pollutants` key mapping pollutant → µg/m³).
            weather: Optional current weather dict.
            fire_summary: Optional fire aggregate dict.
            physics: Optional dict from AISICalculator / RadiationFeedback.
            now: Reference datetime for hour/dow/month features. Defaults
                to live now; training passes the window-end timestamp so
                temporal features are historically correct.

        Returns:
            Flat dictionary of numeric features (JSON serializable).
        """
        features: Dict[str, Any] = {"station_id": station_id}

        # ── Pollutant history + lag statistics ───────────────────────
        series = self._pollutant_series(readings)
        for pollutant in POLLUTANTS:
            values = series.get(pollutant, [])
            features[f"{pollutant}_now"] = self._safe(value(values, -1))

            for window in (24, 48, 72):
                lag = values[-window:] if window <= len(values) else values[:]
                features[f"{pollutant}_mean_{window}h"] = self._safe(self._mean(lag))
                features[f"{pollutant}_max_{window}h"] = self._safe(self._max(lag))
                features[f"{pollutant}_min_{window}h"] = self._safe(self._min(lag))
                features[f"{pollutant}_trend_{window}h"] = self._safe(self._trend(lag))

        # Previous-hour change signal
        pm_now = features.get("pm25_now")
        pm_prev = self._safe(value(series.get("pm25", []), -2))
        features["pm25_delta_1h"] = (
            round(pm_now - pm_prev, 2) if pm_now is not None and pm_prev is not None else 0.0
        )

        # ── Temporal features ────────────────────────────────────────
        now = now or datetime.now()
        features["hour"] = now.hour
        features["day_of_week"] = now.weekday()
        features["month"] = now.month
        features["season"] = SEASON_ENCODE.get(SEASON_MAP.get(now.month, "Winter"), 0)
        features["is_weekend"] = 1 if now.weekday() >= 5 else 0
        features["is_winter"] = 1 if now.month in (11, 12, 1) else 0

        # ── Meteorology ──────────────────────────────────────────────
        weather = weather or {}
        features.update({
            "temperature_c": self._safe(weather.get("temperature_c"), default=None),
            "relative_humidity": self._safe(weather.get("relative_humidity"), default=None),
            "wind_speed_ms": self._safe(weather.get("wind_speed_ms"), default=None),
            "wind_direction_deg": self._safe(weather.get("wind_direction_deg"), default=None),
            "pressure_hpa": self._safe(weather.get("pressure_hpa"), default=None),
            "cloud_cover_pct": self._safe(weather.get("cloud_cover_pct"), default=None),
        })

        # ── Fire / biomass features ──────────────────────────────────
        fire_summary = fire_summary or {}
        features.update({
            "fire_count": float(fire_summary.get("total_fires", 0) or 0),
            "fire_total_frp": float(fire_summary.get("total_frp", 0) or 0),
            "fire_nearest_km": self._safe(fire_summary.get("nearest_km"), default=999.0),
        })

        # ── Physics-derived features ─────────────────────────────────
        physics = physics or {}
        features.update({
            "aisi": self._safe(physics.get("aisi"), default=2.0),
            "pbl_height_m": self._safe(
                (physics.get("pbl") or {}).get("pbl_height_m"), default=700.0
            ),
            "inversion_strength": self._safe(
                (physics.get("pbl") or {}).get("inversion_strength_k"), default=0.0
            ),
            "aod_550nm": self._safe(physics.get("aod_550nm"), default=0.5),
            "surface_forcing": self._safe(physics.get("surface_forcing_wm2"), default=-40.0),
        })

        # Replace None with null-safe code (0.0) for model consumption
        features = {
            k: (v if v is not None else 0.0)
            for k, v in features.items()
            if k != "station_id"
        }
        features["station_id"] = station_id
        return features

    def build_sequence(
        self,
        readings: List[Dict],
        window: Optional[int] = None,
    ) -> List[List[Optional[float]]]:
        """
        Build a 2D sequence window of pollutants for sequence models.

        Returns:
            List of window-length sequences, each an array of
            [pm25, pm10, no2, o3, co] values (in chronological order).
        """
        window = window or self.history_window
        series = self._pollutant_series(readings)
        seq = []

        for i in range(len(readings)):
            row = []
            for p in POLLUTANTS:
                vals = series.get(p, [])
                row.append(self._safe(value(vals, i)))
            seq.append(row)

        return seq[-window:]

    # ── Helpers ──────────────────────────────────────────────────────

    def _pollutant_series(self, readings: List[Dict]) -> Dict[str, List]:
        """Extract per-pollutant chronologically ordered series."""
        series = {p: [] for p in POLLUTANTS}
        for reading in readings:
            pollutants = reading.get("pollutants", {})
            for p in POLLUTANTS:
                v = pollutants.get(p)
                try:
                    series[p].append(float(v) if v is not None else None)
                except (TypeError, ValueError):
                    series[p].append(None)
        return series

    def _mean(self, values) -> Optional[float]:
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        return round(sum(clean) / len(clean), 2)

    def _max(self, values) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return round(max(clean), 2) if clean else None

    def _min(self, values) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return round(min(clean), 2) if clean else None

    def _trend(self, values) -> Optional[float]:
        clean = [v for v in values if v is not None]
        if len(clean) < 4:
            return None
        return round((clean[-1] - clean[0]) / max(len(clean), 1), 2)

    def _safe(self, val, default=None):
        if val is None:
            return default
        try:
            val = float(val)
            return val if math.isfinite(val) else default
        except (TypeError, ValueError):
            return default


def value(values: List, index: int) -> Optional[float]:
    """Return value at index counting from the end (negative index)."""
    if not values:
        return None
    idx = len(values) + index if index < 0 else index
    if idx < 0 or idx >= len(values):
        return None
    return values[idx]