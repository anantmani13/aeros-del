"""
Meteorological Data Client — Open-Meteo (Free, Keyless)

Fetches temperature, relative humidity, wind speed/direction, pressure,
and PBL-relevant parameters for Delhi NCR stations.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WeatherClient:
    """
    Async client for Open-Meteo API (free, no API key required).

    Fetches both current and forecast meteorological data used for:
    - PBL height estimation
    - AISI calculation
    - Wind field for plume transport
    - Feature engineering for ML models
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Open-Meteo variable names for our requirements
    HOURLY_VARS = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "surface_pressure",
        "cloud_cover",
        "visibility",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "temperature_850hPa",
        "wind_speed_850hPa",
        "wind_direction_850hPa",
        "geopotential_height_850hPa",
        "boundary_layer_height",  # ERA5-based PBL height
    ]

    def __init__(self):
        self._http_client = None

    async def _get_http_client(self):
        """Lazy-initialize HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=30.0)
            except ImportError:
                self._http_client = "unavailable"
        return self._http_client

    async def get_current_weather(
        self,
        latitude: float,
        longitude: float,
    ) -> Optional[Dict]:
        """
        Fetch current weather conditions for a location.

        Args:
            latitude: Station latitude
            longitude: Station longitude

        Returns:
            Dict with current meteorological parameters
        """
        client = await self._get_http_client()
        if client == "unavailable":
            return self._generate_fallback_weather()

        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "surface_pressure",
                    "cloud_cover",
                ]),
                "timezone": "Asia/Kolkata",
            }

            response = await client.get(self.BASE_URL, params=params)

            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})
                return {
                    "temperature_c": current.get("temperature_2m"),
                    "temperature_k": (current.get("temperature_2m", 25) + 273.15)
                        if current.get("temperature_2m") is not None else None,
                    "relative_humidity": current.get("relative_humidity_2m"),
                    "wind_speed_ms": current.get("wind_speed_10m"),
                    "wind_direction_deg": current.get("wind_direction_10m"),
                    "pressure_hpa": current.get("surface_pressure"),
                    "cloud_cover_pct": current.get("cloud_cover"),
                    "timestamp": current.get("time"),
                }
            else:
                logger.warning(f"Open-Meteo returned {response.status_code}")
                return self._generate_fallback_weather()

        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
            return self._generate_fallback_weather()

    async def get_hourly_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 3,
    ) -> Optional[Dict]:
        """
        Fetch hourly forecast data for ML features and physics engine.

        Args:
            latitude: Station latitude
            longitude: Station longitude
            forecast_days: Number of forecast days (1-7)

        Returns:
            Dict with hourly arrays for each meteorological variable
        """
        client = await self._get_http_client()
        if client == "unavailable":
            return None

        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ",".join(self.HOURLY_VARS),
                "forecast_days": forecast_days,
                "timezone": "Asia/Kolkata",
            }

            response = await client.get(self.BASE_URL, params=params)

            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})
                return self._parse_hourly(hourly)

        except Exception as e:
            logger.error(f"Forecast fetch failed: {e}")

        return None

    async def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> Optional[Dict]:
        """
        Fetch historical weather data for model training.

        Uses Open-Meteo Historical API (ERA5 reanalysis).

        Args:
            latitude: Station latitude
            longitude: Station longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dict with hourly arrays for the historical period
        """
        client = await self._get_http_client()
        if client == "unavailable":
            return None

        historical_url = "https://archive-api.open-meteo.com/v1/archive"

        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "surface_pressure",
                    "cloud_cover",
                    "shortwave_radiation",
                ]),
                "timezone": "Asia/Kolkata",
            }

            response = await client.get(historical_url, params=params)

            if response.status_code == 200:
                data = response.json()
                return self._parse_hourly(data.get("hourly", {}))

        except Exception as e:
            logger.error(f"Historical weather fetch failed: {e}")

        return None

    def _parse_hourly(self, hourly: Dict) -> Dict:
        """Parse Open-Meteo hourly response into clean dict."""
        parsed = {
            "timestamps": hourly.get("time", []),
            "temperature_c": hourly.get("temperature_2m", []),
            "relative_humidity": hourly.get("relative_humidity_2m", []),
            "wind_speed_ms": hourly.get("wind_speed_10m", []),
            "wind_direction_deg": hourly.get("wind_direction_10m", []),
            "wind_gusts_ms": hourly.get("wind_gusts_10m", []),
            "pressure_hpa": hourly.get("surface_pressure", []),
            "cloud_cover_pct": hourly.get("cloud_cover", []),
            "visibility_m": hourly.get("visibility", []),
            "shortwave_radiation": hourly.get("shortwave_radiation", []),
            "direct_irradiance": hourly.get("direct_normal_irradiance", []),
            "pbl_height_m": hourly.get("boundary_layer_height", []),
            "temp_850hpa": hourly.get("temperature_850hPa", []),
            "wind_850hpa_speed": hourly.get("wind_speed_850hPa", []),
            "wind_850hpa_dir": hourly.get("wind_direction_850hPa", []),
        }

        # Compute derived variables
        if parsed["temperature_c"]:
            parsed["temperature_k"] = [
                t + 273.15 if t is not None else None
                for t in parsed["temperature_c"]
            ]

        # Compute u,v wind components from speed and direction
        if parsed["wind_speed_ms"] and parsed["wind_direction_deg"]:
            import math
            parsed["u_wind"] = []
            parsed["v_wind"] = []
            for spd, dirn in zip(
                parsed["wind_speed_ms"], parsed["wind_direction_deg"]
            ):
                if spd is not None and dirn is not None:
                    rad = math.radians(dirn)
                    parsed["u_wind"].append(-spd * math.sin(rad))
                    parsed["v_wind"].append(-spd * math.cos(rad))
                else:
                    parsed["u_wind"].append(0.0)
                    parsed["v_wind"].append(0.0)

        return parsed

    def _generate_fallback_weather(self) -> Dict:
        """Deterministic Delhi defaults when the API is down.

        Deliberately NOT random: jitter here used to swing AISI by points
        between refresh cycles. Values follow a smooth September diurnal.
        """
        try:
            from zoneinfo import ZoneInfo
            hour = datetime.now(timezone.utc).astimezone(
                ZoneInfo("Asia/Kolkata")).hour
        except Exception:
            hour = datetime.now(timezone.utc).hour
        import math
        day_w = 0.5 * (1.0 - math.cos((hour - 14) * math.pi / 12.0))
        return {
            "temperature_c": round(26.0 + 6.0 * day_w, 1),
            "temperature_k": round(299.15 + 6.0 * day_w, 1),
            "relative_humidity": round(75.0 - 25.0 * day_w, 1),
            "wind_speed_ms": 2.5,
            "wind_direction_deg": 290.0,
            "pressure_hpa": 1008.0,
            "cloud_cover_pct": 20.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "fallback",
        }

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and self._http_client != "unavailable":
            await self._http_client.aclose()
            self._http_client = None
