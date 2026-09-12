"""
NASA FIRMS Fire Hotspot Client

Fetches active fire data from NASA FIRMS (Fire Information for Resource
Management System) including MODIS and VIIRS satellite detections.
Used for stubble burning detection and plume source identification.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FireClient:
    """
    Async client for NASA FIRMS API.

    Fetches active fire hotspots relevant to Delhi NCR,
    including Punjab/Haryana stubble burning region.
    """

    FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    FIRMS_MAP_KEY_URL = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"

    # Extended bounding box to capture Punjab/Haryana fires
    FIRE_BBOX = {
        "lat_min": 27.0,   # South of Delhi
        "lat_max": 32.5,   # North Punjab
        "lon_min": 74.5,   # West Haryana
        "lon_max": 78.0,   # East UP border
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "DEMO_KEY"
        self._http_client = None

    async def _get_http_client(self):
        """Lazy-initialize HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=60.0)
            except ImportError:
                self._http_client = "unavailable"
        return self._http_client

    async def get_active_fires(
        self,
        days_back: int = 2,
        source: str = "VIIRS_SNPP_NRT",
    ) -> List[Dict]:
        """
        Fetch active fire hotspots in the Punjab-Haryana-Delhi region.

        Args:
            days_back: Number of days of data (1-10)
            source: Satellite source (VIIRS_SNPP_NRT, MODIS_NRT, VIIRS_NOAA20_NRT)

        Returns:
            List of fire hotspot dicts with lat, lon, FRP, confidence, etc.
        """
        client = await self._get_http_client()
        if client == "unavailable":
            return self._generate_demo_fires()

        try:
            # FIRMS API uses MAP_KEY for authentication
            bbox_str = (
                f"{self.FIRE_BBOX['lon_min']},"
                f"{self.FIRE_BBOX['lat_min']},"
                f"{self.FIRE_BBOX['lon_max']},"
                f"{self.FIRE_BBOX['lat_max']}"
            )

            url = (
                f"{self.FIRMS_BASE_URL}/{self.api_key}"
                f"/{source}/{bbox_str}/{days_back}"
            )

            response = await client.get(url)

            if response.status_code == 200:
                return self._parse_csv_response(response.text)
            else:
                logger.warning(f"FIRMS API returned {response.status_code}")
                return self._generate_demo_fires()

        except Exception as e:
            logger.error(f"Fire data fetch failed: {e}")
            return self._generate_demo_fires()

    def _parse_csv_response(self, csv_text: str) -> List[Dict]:
        """Parse FIRMS CSV response into list of fire dicts."""
        fires = []
        lines = csv_text.strip().split("\n")

        if len(lines) < 2:
            return fires

        headers = lines[0].lower().split(",")

        for line in lines[1:]:
            values = line.split(",")
            if len(values) != len(headers):
                continue

            row = dict(zip(headers, values))

            try:
                fire = {
                    "latitude": float(row.get("latitude", 0)),
                    "longitude": float(row.get("longitude", 0)),
                    "brightness": float(row.get("bright_ti4", 0)
                        or row.get("brightness", 0)),
                    "frp": float(row.get("frp", 0)),
                    "confidence": row.get("confidence", "nominal"),
                    "acq_date": row.get("acq_date", ""),
                    "acq_time": row.get("acq_time", ""),
                    "satellite": row.get("satellite", ""),
                    "daynight": row.get("daynight", "D"),
                }

                # Calculate distance from Delhi center
                fire["distance_to_delhi_km"] = self._haversine_distance(
                    fire["latitude"], fire["longitude"],
                    28.6139, 77.2090,
                )

                # Classify fire region
                fire["region"] = self._classify_region(
                    fire["latitude"], fire["longitude"]
                )

                fires.append(fire)

            except (ValueError, KeyError) as e:
                continue

        logger.info(f"Parsed {len(fires)} fire hotspots")
        return fires

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate great-circle distance between two points in km."""
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
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 2)

    def _classify_region(self, lat: float, lon: float) -> str:
        """Classify fire location into region."""
        if lat >= 30.0 and lon < 76.5:
            return "Punjab"
        elif lat >= 29.0 and lon < 77.0:
            return "Haryana"
        elif lat >= 28.3 and lat <= 28.9:
            return "Delhi NCR"
        elif lon > 77.5:
            return "Western UP"
        else:
            return "Other"

    def aggregate_fire_stats(self, fires: List[Dict]) -> Dict:
        """
        Aggregate fire statistics for dashboard display.

        Returns:
            Dict with total counts, regional breakdown, total FRP, etc.
        """
        if not fires:
            return {
                "total_fires": 0,
                "total_frp_mw": 0,
                "regions": {},
                "avg_distance_km": 0,
                "nearest_fire_km": 0,
                "high_confidence_count": 0,
            }

        regions = {}
        total_frp = 0
        distances = []
        high_conf = 0

        for f in fires:
            region = f.get("region", "Other")
            regions[region] = regions.get(region, 0) + 1
            total_frp += f.get("frp", 0)
            distances.append(f.get("distance_to_delhi_km", 0))
            if str(f.get("confidence", "")).lower() in ("high", "h", "100"):
                high_conf += 1

        return {
            "total_fires": len(fires),
            "total_frp_mw": round(total_frp, 1),
            "regions": regions,
            "avg_distance_km": round(sum(distances) / len(distances), 1),
            "nearest_fire_km": round(min(distances), 1) if distances else 0,
            "high_confidence_count": high_conf,
        }

    def _generate_demo_fires(self) -> List[Dict]:
        """Generate realistic demo fire data when API is unavailable."""
        import random

        demo_fires = []
        # Simulate Punjab stubble burning season
        punjab_lats = [30.5, 30.8, 31.0, 31.2, 30.3, 30.7, 31.5]
        punjab_lons = [75.0, 75.5, 74.8, 75.2, 76.0, 75.8, 74.5]

        for lat, lon in zip(punjab_lats, punjab_lons):
            demo_fires.append({
                "latitude": lat + random.uniform(-0.2, 0.2),
                "longitude": lon + random.uniform(-0.2, 0.2),
                "brightness": 310 + random.uniform(0, 30),
                "frp": random.uniform(5, 80),
                "confidence": random.choice(["high", "nominal", "low"]),
                "acq_date": datetime.now().strftime("%Y-%m-%d"),
                "acq_time": f"{random.randint(6, 18):02d}{random.randint(0, 59):02d}",
                "satellite": "VIIRS",
                "daynight": "D",
                "distance_to_delhi_km": self._haversine_distance(
                    lat, lon, 28.6139, 77.2090
                ),
                "region": "Punjab",
            })

        # Add some Haryana fires
        for _ in range(3):
            lat = 29.0 + random.uniform(0, 1.0)
            lon = 76.0 + random.uniform(0, 0.8)
            demo_fires.append({
                "latitude": lat,
                "longitude": lon,
                "brightness": 305 + random.uniform(0, 20),
                "frp": random.uniform(3, 40),
                "confidence": "nominal",
                "acq_date": datetime.now().strftime("%Y-%m-%d"),
                "acq_time": "1200",
                "satellite": "VIIRS",
                "daynight": "D",
                "distance_to_delhi_km": self._haversine_distance(
                    lat, lon, 28.6139, 77.2090
                ),
                "region": "Haryana",
            })

        return demo_fires

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and self._http_client != "unavailable":
            await self._http_client.aclose()
            self._http_client = None
