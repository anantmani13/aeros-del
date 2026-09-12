"""
WAQI (World Air Quality Index, aqicn.org) Client — SECONDARY Source.

Role: backup pollutant readings + independent forecast for cross-check.
OpenAQ/CPCB stays primary (raw monitor data = ground truth story).
Chain: OpenAQ -> WAQI -> demo. Any failure here returns []/None and the
pipeline continues — this source can only help, never break.

Needs WAQI_API_KEY in .env (free token from https://aqicn.org/api/).
Without a key every method returns empty (no crash, no demo fakery).
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_URL = "https://api.waqi.info"

# WAQI iaqi keys -> our internal pollutant names.
# NOTE: WAQI v-values for gases are already µg/m³ except co (mg/m³ OK).
IAQI_MAP = {
    "pm25": "pm25",
    "pm10": "pm10",
    "no2": "no2",
    "so2": "so2",
    "o3": "o3",
    "co": "co",
}

BOUNDS = {
    "pm25": (0, 1500), "pm10": (0, 2000), "no2": (0, 800),
    "so2": (0, 1000), "o3": (0, 600), "co": (0, 50),
}


class WAQIClient:
    """Async client for the WAQI station/forecast API."""

    def __init__(self, api_key: Optional[str] = None,
                 rate_limit: float = 1.0):
        # Free tokens are rate-limited (~per-second); stay polite.
        self.api_key = (api_key or "").strip()
        self.rate_limit = rate_limit
        self._last = 0.0
        self._http = None

    async def _get(self, path: str, params: Dict) -> Optional[Dict]:
        if not self.api_key:
            return None
        elapsed = time.time() - self._last
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        try:
            import httpx
            if self._http is None:
                self._http = httpx.AsyncClient(timeout=30.0)
            self._last = time.time()
            r = await self._http.get(f"{BASE_URL}{path}",
                                     params={**params, "token": self.api_key})
            if r.status_code != 200:
                logger.warning("WAQI %s -> %s", path, r.status_code)
                return None
            body = r.json()
            if body.get("status") != "ok":
                logger.warning("WAQI %s: %s", path, body.get("data"))
                return None
            return body.get("data")
        except Exception as e:
            logger.warning("WAQI request failed: %s", e)
            return None

    async def get_station_feed(self, latitude: float,
                               longitude: float) -> Optional[Dict]:
        """Nearest-station feed: current iaqi + daily forecast.

        Returns dict {station, uid, timestamp, pollutants, aqi, forecast}
        or None on any failure/missing key.
        """
        data = await self._get(f"/feed/geo:{latitude};{longitude}/", {})
        if not data:
            return None
        pollutants: Dict[str, float] = {}
        for k, v in (data.get("iaqi") or {}).items():
            internal = IAQI_MAP.get(k)
            if not internal:
                continue
            try:
                val = float((v or {}).get("v"))
            except (TypeError, ValueError):
                continue
            lo, hi = BOUNDS[internal]
            if lo <= val <= hi:
                pollutants[internal] = val
        if not pollutants:
            return None
        city = data.get("city") or {}
        t = (data.get("time") or {})
        return {
            "station": city.get("name", "WAQI station"),
            "uid": data.get("idx"),
            "latitude": (city.get("geo") or [latitude, longitude])[0],
            "longitude": (city.get("geo") or [latitude, longitude])[1],
            "timestamp": t.get("iso") or datetime.now(timezone.utc).isoformat(),
            "pollutants": pollutants,
            "aqi": data.get("aqi"),
            "forecast": self._parse_forecast(data.get("forecast") or {}),
            "source": "waqi",
        }

    @staticmethod
    def _parse_forecast(fc: Dict) -> Dict[str, List[Dict]]:
        """Daily {avg,max,min,day} series per pollutant (their forecast)."""
        out: Dict[str, List[Dict]] = {}
        for pol, days in (fc.get("daily") or {}).items():
            if pol not in IAQI_MAP:
                continue
            series = []
            for d in days or []:
                try:
                    series.append({
                        "day": d.get("day"),
                        "avg": float(d.get("avg")),
                        "max": float(d.get("max")),
                        "min": float(d.get("min")),
                    })
                except (TypeError, ValueError):
                    continue
            if series:
                out[IAQI_MAP[pol]] = series
        return out

    async def close(self):
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None
