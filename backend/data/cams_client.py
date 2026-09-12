"""
CAMS Live-Model Client — Open-Meteo Air Quality (Free, Keyless)

Keyless live fallback that keeps every station showing current-hour data
when the CPCB observation feed stalls upstream (e.g. OpenAQ's CPCB
ingestion lagging 30h+ while only a handful of private monitors stay
fresh — Sept 2026 incident).

This is CAMS model nowcast output, NOT station observations. Every datum
is labeled source="cams" and the service layer never persists model
values as observed history. No API key, same provider family as the
existing weather client.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CAMSClient:
    """Async client for the Open-Meteo air-quality (CAMS) API."""

    BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    HOURLY_VARS = [
        "pm2_5",
        "pm10",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "carbon_monoxide",
    ]
    VAR_MAP = {
        "pm2_5": "pm25",
        "pm10": "pm10",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3",
        "carbon_monoxide": "co",
    }
    # Internal units: µg/m³ except co (mg/m³); Open-Meteo reports co in
    # µg/m³, converted on parse.
    BOUNDS = {
        "pm25": (0, 1500),
        "pm10": (0, 2000),
        "no2": (0, 800),
        "so2": (0, 1000),
        "o3": (0, 600),
        "co": (0, 50),
    }

    def __init__(self, max_concurrency: int = 8):
        self.max_concurrency = max(1, max_concurrency)
        self._http_client = None

    async def _get_http_client(self):
        """Lazy-initialize the HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=30.0)
            except ImportError:
                self._http_client = "unavailable"
        return self._http_client

    @staticmethod
    def _slot_index(times: List[str],
                    now: Optional[datetime] = None) -> int:
        """Index of the latest hourly slot at or before now (UTC)."""
        ref = now or datetime.now(timezone.utc)
        best = 0
        for i, ts in enumerate(times):
            try:
                dt = datetime.strptime(str(ts)[:16], "%Y-%m-%dT%H:%M")
                dt = dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if dt <= ref:
                best = i
        return best

    def _parse_current(self, data: Optional[Dict],
                       now: Optional[datetime] = None) -> Optional[Dict]:
        """Extract the current-hour datum from an hourly response."""
        hourly = (data or {}).get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return None
        idx = self._slot_index(times, now=now)
        pollutants: Dict[str, float] = {}
        for var, internal in self.VAR_MAP.items():
            series = hourly.get(var) or []
            if idx >= len(series) or series[idx] is None:
                continue
            try:
                value = float(series[idx])
            except (TypeError, ValueError):
                continue
            if internal == "co":
                value = value / 1000.0  # µg/m³ → mg/m³
            lo, hi = self.BOUNDS[internal]
            if lo <= value <= hi:
                pollutants[internal] = round(value, 2)
        if "pm25" not in pollutants and "pm10" not in pollutants:
            return None  # need at least PM to be useful
        slot = str(times[idx])[:16]  # "YYYY-MM-DDTHH:MM" UTC
        return {
            "pollutants": pollutants,
            "timestamp": f"{slot}:00+00:00",
            "source": "cams",
        }

    async def get_current_aq(
        self,
        latitude: float,
        longitude: float,
    ) -> Optional[Dict]:
        """
        Fetch the current-hour CAMS nowcast for one coordinate.

        Returns:
            Dict with pollutants / timestamp / source="cams", or None.
        """
        client = await self._get_http_client()
        if client == "unavailable":
            return None
        try:
            resp = await client.get(self.BASE_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ",".join(self.HOURLY_VARS),
                "timezone": "UTC",
                "forecast_days": 1,
            })
            if resp.status_code != 200:
                logger.warning("CAMS API error %s for %s,%s",
                               resp.status_code, latitude, longitude)
                return None
            return self._parse_current(resp.json())
        except Exception as e:
            logger.warning("CAMS fetch failed (%s,%s): %s",
                           latitude, longitude, e)
            return None

    async def get_current_batch(
        self,
        coords: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Dict]:
        """
        Fetch current-hour CAMS nowcasts for many stations concurrently.

        Args:
            coords: station_id -> (latitude, longitude)

        Returns:
            station_id -> datum (only successful fetches included)
        """
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one(sid: str, lat: float, lon: float):
            async with sem:
                datum = await self.get_current_aq(lat, lon)
                return sid, datum

        out: Dict[str, Dict] = {}
        for sid, datum in await asyncio.gather(
            *(_one(sid, lat, lon) for sid, (lat, lon) in coords.items())
        ):
            if datum and datum.get("pollutants"):
                out[sid] = datum
        return out

    async def close(self):
        """Close the HTTP client."""
        client, self._http_client = self._http_client, None
        if client and client != "unavailable":
            aclose = getattr(client, "aclose", None)
            if callable(aclose):
                await aclose()
