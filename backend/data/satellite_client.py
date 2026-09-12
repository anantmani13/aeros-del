"""
Satellite AOD Data Client

Fetches Aerosol Optical Depth (AOD) data from MODIS (via AppEEARS)
and optionally INSAT-3D for aerosol-radiation coupling calculations.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SatelliteClient:
    """
    Satellite AOD data client.

    Fetches MODIS AOD data. Falls back to empirical PM2.5-to-AOD
    estimation when satellite data is unavailable.
    """

    APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

    def __init__(self, earthdata_token: Optional[str] = None):
        self.token = earthdata_token
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

    async def get_modis_aod(
        self,
        latitude: float,
        longitude: float,
        date: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Fetch MODIS AOD for a location and date.

        Args:
            latitude: Station latitude
            longitude: Station longitude
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Dict with AOD values, or None if unavailable
        """
        client = await self._get_http_client()
        if client == "unavailable" or not self.token:
            return self._estimate_aod_from_defaults(latitude, longitude)

        try:
            # AppEEARS point sample request
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")

            response = await client.get(
                f"{self.APPEEARS_BASE}/point",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "date": date,
                    "product": "MCD19A2.061",  # MODIS AOD product
                    "layer": "Optical_Depth_047",
                },
                headers={"Authorization": f"Bearer {self.token}"},
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "aod_550nm": data.get("value"),
                    "quality": data.get("quality", "good"),
                    "source": "MODIS_MCD19A2",
                    "timestamp": date,
                }

        except Exception as e:
            logger.error(f"MODIS AOD fetch failed: {e}")

        return self._estimate_aod_from_defaults(latitude, longitude)

    def estimate_aod_from_pm25(self, pm25: float) -> Dict:
        """
        Estimate AOD from surface PM2.5 using empirical relationship.

        Based on Delhi-specific MODIS validation studies:
        AOD ≈ 0.003 * PM2.5 + 0.15 (for Delhi NCR, r² ≈ 0.65)

        Args:
            pm25: Surface PM2.5 concentration (µg/m³)

        Returns:
            Dict with estimated AOD and metadata
        """
        # Use the formulas module for physics-based estimate
        from backend.formulas.radiation_formulas import aod_from_pm25

        aod_physics = aod_from_pm25(pm25)

        # Also compute empirical linear estimate
        aod_empirical = 0.003 * pm25 + 0.15

        # Blend both estimates
        aod_blended = 0.6 * aod_physics + 0.4 * aod_empirical

        return {
            "aod_550nm": round(aod_blended, 3),
            "aod_physics": round(aod_physics, 3),
            "aod_empirical": round(aod_empirical, 3),
            "source": "estimated_from_pm25",
            "pm25_input": pm25,
        }

    def _estimate_aod_from_defaults(
        self,
        latitude: float,
        longitude: float,
    ) -> Dict:
        """Return seasonal default AOD for Delhi NCR."""
        import random

        # Delhi NCR seasonal average AOD at 550nm
        month = datetime.now().month
        if month in [11, 12, 1]:     # Winter — high haze
            base_aod = 0.8
        elif month in [4, 5, 6]:     # Summer — dust
            base_aod = 0.5
        elif month in [7, 8, 9]:     # Monsoon — washout
            base_aod = 0.3
        else:                         # Post-monsoon / transition
            base_aod = 0.6

        aod = base_aod + random.uniform(-0.1, 0.2)

        return {
            "aod_550nm": round(max(0.05, aod), 3),
            "source": "seasonal_default",
            "quality": "estimated",
        }

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and self._http_client != "unavailable":
            await self._http_client.aclose()
            self._http_client = None
