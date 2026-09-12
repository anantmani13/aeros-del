"""
OpenAQ Data Client — PRIMARY Air Quality Data Source

Fetches hourly PM2.5, PM10, NO2, SO2, O3, CO for ~40 Delhi NCR stations
from the OpenAQ v3 API. Includes rate limiting, retry logic, caching,
historical data retrieval, and validation against physical bounds.
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Physical bounds for validation ───────────────────────────────────
POLLUTANT_BOUNDS = {
    "pm25": (0, 1500),     # µg/m³
    "pm10": (0, 2000),     # µg/m³
    "no2": (0, 800),       # µg/m³
    "so2": (0, 1000),      # µg/m³
    "o3": (0, 600),        # µg/m³
    "co": (0, 50),         # mg/m³
}

# Map OpenAQ parameter names to our internal names
OPENAQ_PARAM_MAP = {
    "pm25": "pm25",
    "pm10": "pm10",
    "no2": "no2",
    "so2": "so2",
    "o3": "o3",
    "co": "co",
}


@dataclass
class StationReading:
    """Represents a single station's pollutant readings at a point in time."""

    station_id: str
    station_name: str
    latitude: float
    longitude: float
    timestamp: str
    pollutants: Dict[str, Optional[float]]
    source: str = "openaq"
    provider: str = "unknown"


class OpenAQClient:
    """
    Async client for OpenAQ API v3.

    Features:
    - Rate-limited requests (configurable, default 0.1s between calls)
    - Automatic retry with exponential backoff
    - In-memory cache for recent readings
    - Bounding box queries for Delhi NCR
    - Historical data retrieval for model training
    - Physical bounds validation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openaq.org/v3",
        rate_limit: float = 0.1,
        cache_ttl: int = 300,
        max_concurrency: int = 4,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.cache_ttl = cache_ttl
        self.max_concurrency = max(1, max_concurrency)
        self._last_request_time = 0.0
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._sensor_param_map: Dict[int, str] = {}
        self._pace_lock: Optional[asyncio.Lock] = None
        self._http_client = None

        # Delhi NCR bounding box
        self.bbox = {
            "lat_min": 28.30,
            "lat_max": 28.90,
            "lon_min": 76.80,
            "lon_max": 77.50,
        }

    async def _get_http_client(self):
        """Lazy-initialize the HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                headers = {"Accept": "application/json"}
                if self.api_key:
                    headers["X-API-Key"] = self.api_key
                self._http_client = httpx.AsyncClient(
                    headers=headers,
                    timeout=30.0,
                )
            except ImportError:
                import aiohttp
                self._http_client = "aiohttp"
        return self._http_client

    async def _rate_limited_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        max_retries: int = 3,
    ) -> Optional[Dict]:
        """Make a rate-limited API request with retry logic."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(max_retries):
            try:
                self._last_request_time = time.time()
                client = await self._get_http_client()

                if isinstance(client, str):
                    # aiohttp fallback
                    import aiohttp
                    headers = {"Accept": "application/json"}
                    if self.api_key:
                        headers["X-API-Key"] = self.api_key
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                        ) as resp:
                            if resp.status == 200:
                                return await resp.json()
                            elif resp.status == 429:
                                wait = 2 ** (attempt + 1)
                                logger.warning(f"Rate limited, waiting {wait}s")
                                await asyncio.sleep(wait)
                                continue
                            else:
                                logger.warning(
                                    f"OpenAQ API error {resp.status}: {await resp.text()}"
                                )
                else:
                    response = await client.get(url, params=params)
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 429:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        logger.warning(
                            f"OpenAQ API error {response.status_code}: {response.text}"
                        )

            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return None

    def _validate_value(self, param: str, value: float) -> bool:
        """Validate a pollutant value against physical bounds."""
        bounds = POLLUTANT_BOUNDS.get(param)
        if bounds is None:
            return True
        return bounds[0] <= value <= bounds[1]

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get a value from cache if still valid."""
        if key in self._cache:
            age = time.time() - self._cache_timestamps.get(key, 0)
            if age < self.cache_ttl:
                return self._cache[key]
            else:
                del self._cache[key]
                del self._cache_timestamps[key]
        return None

    def _set_cached(self, key: str, value: Any):
        """Store a value in cache."""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()

    async def get_locations_in_delhi(
        self, force_fresh: bool = False,
    ) -> List[Dict]:
        """
        Fetch all monitoring locations in Delhi NCR bounding box.

        Args:
            force_fresh: Skip the in-memory cache and hit the API
                (manual refresh must always fetch live data).

        Returns:
            List of location dicts with id, name, coordinates, parameters
        """
        cache_key = "locations_delhi"
        if not force_fresh:
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        base_params = {
            "bbox": (f"{self.bbox['lon_min']},{self.bbox['lat_min']},"
                     f"{self.bbox['lon_max']},{self.bbox['lat_max']}"),
            "limit": 100,
            "order_by": "id",
            "sort_order": "asc",
        }

        def _parse_location(loc: Dict) -> Dict:
            # Record sensor -> parameter mapping (v3 /latest only returns
            # sensorsId, so we resolve pollutant names via this map).
            # Also derive the monitored-parameters set from sensors —
            # the top-level "parameters" field is often empty.
            params = set()
            for s in loc.get("sensors") or []:
                param = s.get("parameter")
                if isinstance(param, dict):
                    pname = param.get("name", "").lower()
                    self._sensor_param_map[s.get("id")] = pname
                    params.add(pname)
            prov = loc.get("provider") or {}
            return {
                "id": loc.get("id"),
                "name": loc.get("name", "Unknown"),
                "latitude": loc.get("coordinates", {}).get("latitude"),
                "longitude": loc.get("coordinates", {}).get("longitude"),
                "parameters": sorted(params),
                "provider": (prov.get("name", "unknown")
                             if isinstance(prov, dict)
                             else str(prov or "unknown")),
                "last_updated": (loc.get("datetimeLast") or {}).get("utc")
                    if isinstance(loc.get("datetimeLast"), dict)
                    else loc.get("datetimeLast"),
            }

        locations = []
        # Paginate every page: the API sorts id-ascending, so page 1 is
        # mostly dead archive monitors (2016–2022) while live ones sit on
        # later pages / high ids. Fetching only page 1 starved the
        # /latest fan-out of live locations (Sept 2026 incident: every
        # station rendered 33h-stale CPCB data while fresh monitors on
        # page 2 were never queried).
        for page in range(1, 4):  # max 300 locations; Delhi NCR has ~116
            params = dict(base_params, page=page)
            # Try bounding box first, fall back to coordinates + radius
            data = await self._rate_limited_request("locations", params)

            if (not data or "results" not in data) and page == 1:
                # Fallback: query by coordinates center + radius (max 25km)
                params = {
                    "coordinates": "28.6139,77.2090",
                    "radius": 25000,
                    "limit": 100,
                    "page": 1,
                }
                data = await self._rate_limited_request("locations", params)

            if not data or "results" not in data:
                break
            results = data["results"]
            if not results:
                break
            locations.extend(_parse_location(loc) for loc in results)
            if len(results) < 100:
                break

        if locations:
            # Fresh monitors first (ISO-8601 UTC strings sort
            # chronologically; missing timestamps sink to the end), so the
            # [:N] slice in get_latest_measurements keeps live locations
            # instead of dead archives.
            locations.sort(key=lambda l: l.get("last_updated") or "",
                           reverse=True)
            self._set_cached(cache_key, locations)

        logger.info(f"Found {len(locations)} OpenAQ locations in Delhi NCR")
        return locations

    async def get_latest_measurements(
        self,
        location_id: Optional[int] = None,
        force_fresh: bool = False,
    ) -> List[StationReading]:
        """
        Fetch latest measurements for Delhi NCR stations.

        Args:
            location_id: Specific location ID, or None for all Delhi NCR
            force_fresh: Skip the in-memory cache and hit the API
                (manual refresh must always fetch live data).

        Returns:
            List of StationReading objects
        """
        if location_id:
            cache_key = f"latest_{location_id}"
            if not force_fresh:
                cached = self._get_cached(cache_key)
                if cached:
                    return cached

            data = await self._rate_limited_request(
                f"locations/{location_id}/latest"
            )
            readings = self._parse_latest_response(data)
            self._set_cached(cache_key, readings)
            return readings

        # Fetch all Delhi NCR
        cache_key = "latest_all_delhi"
        if not force_fresh:
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        locations = await self.get_locations_in_delhi(
            force_fresh=force_fresh)

        # Only locations that actually measure PM — limits API calls and
        # skips stations without the pollutants we model. The list arrives
        # fresh-first (see get_locations_in_delhi), so this slice keeps
        # live monitors instead of dead archives.
        locations = [
            loc for loc in locations
            if any(p in ("pm25", "pm10") for p in (loc.get("parameters") or []))
        ][:60]

        all_readings = []

        # Parallel fetch with bounded concurrency: dispatch pacing is
        # serialized under a lock so bursts stay inside OpenAQ's quota,
        # while in-flight requests overlap (startup 60-90s → ~10-20s).
        if self._pace_lock is None:
            self._pace_lock = asyncio.Lock()
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _fetch_one(loc: Dict) -> List[StationReading]:
            async with sem:
                async with self._pace_lock:
                    elapsed = time.time() - self._last_request_time
                    spacing = max(self.rate_limit, 0.35)
                    if elapsed < spacing:
                        await asyncio.sleep(spacing - elapsed)
                    self._last_request_time = time.time()
                loc_data = await self._rate_limited_request(
                    f"locations/{loc['id']}/latest"
                )
                if loc_data:
                    return self._parse_latest_response(loc_data, loc)
                return []

        for readings in await asyncio.gather(
            *(_fetch_one(loc) for loc in locations)
        ):
            all_readings.extend(readings)

        self._set_cached(cache_key, all_readings)
        return all_readings

    def _parse_latest_response(
        self,
        data: Optional[Dict],
        location_info: Optional[Dict] = None,
    ) -> List[StationReading]:
        """Parse OpenAQ latest measurements response into StationReadings."""
        if not data or "results" not in data:
            return []

        readings = []
        results = data["results"]

        # Group by location
        for result in results:
            pollutants = {}
            timestamp = None

            measurements = result.get("measurements", [])
            if isinstance(result, dict) and "parameter" in result:
                measurements = [result]
            elif not measurements and "value" in result:
                # v3 /latest returns each measurement as a bare dict
                # (sensorsId + value + datetime, parameter resolved via map).
                measurements = [result]

            for m in measurements:
                param_name = m.get("parameter", {})
                if isinstance(param_name, dict):
                    param_name = param_name.get("name", "").lower()
                else:
                    param_name = str(param_name).lower()

                # v3 /latest omits the parameter name; resolve it via the
                # sensor->parameter map built when locations were listed.
                if (not param_name or param_name == "none") and m.get("sensorsId"):
                    param_name = (self._sensor_param_map
                                  .get(m.get("sensorsId"), "") or "").lower()

                internal_name = OPENAQ_PARAM_MAP.get(param_name)
                if internal_name and m.get("value") is not None:
                    value = float(m["value"])
                    if self._validate_value(internal_name, value):
                        pollutants[internal_name] = value

                if not timestamp and m.get("datetime"):
                    dt = m["datetime"]
                    if isinstance(dt, dict):
                        timestamp = dt.get("utc") or dt.get("local")
                    else:
                        timestamp = str(dt)

            if pollutants:
                loc = location_info or {}
                coords = result.get("coordinates", {})

                # Drop clearly-stale archive sensors (>= 2019 CPCB heritage)
                stamp = None
                if timestamp:
                    try:
                        stamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                        if stamp < cutoff:
                            continue
                    except (ValueError, TypeError):
                        stamp = None

                readings.append(StationReading(
                    station_id=str(
                        loc.get("id") or result.get("id") or "unknown"
                    ),
                    station_name=loc.get("name")
                        or result.get("name", "Unknown"),
                    latitude=coords.get("latitude")
                        or loc.get("latitude", 0.0),
                    longitude=coords.get("longitude")
                        or loc.get("longitude", 0.0),
                    timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
                    pollutants=pollutants,
                    source="openaq",
                    provider=loc.get("provider", "unknown") or "unknown",
                ))

        return readings

    async def get_sensor_measurements(
        self,
        sensor_id: int,
        datetime_from: str,
        datetime_to: str,
        limit: int = 1000,
        max_pages: int = 25,
    ) -> List[Dict]:
        """
        Fetch historical measurements for one sensor (model training /
        history backfill).

        Hits `GET /v3/sensors/{id}/measurements` with `datetime_from` /
        `datetime_to` (the `date_from` variant is silently ignored by the
        API and returns unfiltered archive data — verified Sept 2026).

        Args:
            sensor_id: OpenAQ sensor ID (from location["sensors"])
            datetime_from: Start datetime (ISO format, UTC)
            datetime_to: End datetime (ISO format, UTC)
            limit: Results per page (max 1000)
            max_pages: Safety cap on pagination

        Returns:
            List of dicts with utc datetime string, value, parameter name
        """
        all_measurements: List[Dict] = []
        page = 1

        while page <= max_pages:
            params = {
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": limit,
                "page": page,
            }

            data = await self._rate_limited_request(
                f"sensors/{sensor_id}/measurements", params
            )

            if not data or "results" not in data:
                break

            results = data["results"]
            if not results:
                break

            for m in results:
                value = m.get("value")
                param = (m.get("parameter") or {}).get("name", "").lower()
                period = m.get("period") or {}
                dt = period.get("datetimeFrom") or {}
                utc = dt.get("utc") if isinstance(dt, dict) else None
                if value is None or not utc:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if self._validate_value(param, value):
                    all_measurements.append({
                        "utc": utc,
                        "value": value,
                        "parameter": param,
                    })

            if len(results) < limit:
                break
            page += 1

        logger.info(
            f"Retrieved {len(all_measurements)} measurements "
            f"for sensor {sensor_id}"
        )
        return all_measurements

    async def get_historical_measurements(
        self,
        location_id: int,
        date_from: str,
        date_to: str,
        parameter: str = "pm25",
        limit: int = 1000,
    ) -> List[Dict]:
        """
        Fetch historical measurements for model training (legacy wrapper).

        Resolves the location's sensor for `parameter`, then delegates to
        :meth:`get_sensor_measurements`.
        """
        data = await self._rate_limited_request(f"locations/{location_id}")
        sensor_id = None
        results = (data or {}).get("results", [])
        loc = results[0] if results else data
        for s in ((loc or {}).get("sensors") or []):
            if ((s.get("parameter") or {}).get("name", "").lower()
                    == parameter.lower()):
                sensor_id = s.get("id")
                break
        if sensor_id is None:
            logger.warning(f"No {parameter} sensor at location {location_id}")
            return []
        return await self.get_sensor_measurements(
            sensor_id, date_from, date_to, limit=limit,
        )

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and not isinstance(self._http_client, str):
            await self._http_client.aclose()
            self._http_client = None
