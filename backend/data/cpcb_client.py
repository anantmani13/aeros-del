"""
CPCB Data Client — Fallback Only

Legacy client for fetching air quality data directly from CPCB.
Used only when OpenAQ is unavailable. Outputs the same format as OpenAQClient.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CPCBClient:
    """
    CPCB data client (fallback).

    Retained for compatibility. The official CPCB API is frequently
    restricted/unreliable. Use OpenAQClient as primary source.
    """

    def __init__(self):
        self.base_url = "https://app.cpcbccr.com/ccr_docs/ccr_data"
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

    async def get_station_data(
        self,
        station_id: str,
    ) -> Optional[Dict]:
        """
        Fetch latest data for a CPCB station.

        Args:
            station_id: CPCB station identifier

        Returns:
            Dict with pollutant readings, or None if unavailable
        """
        client = await self._get_http_client()
        if client == "unavailable":
            logger.warning("HTTP client unavailable for CPCB fallback")
            return None

        try:
            # CPCB API endpoint (may require session/token)
            response = await client.get(
                f"{self.base_url}",
                params={
                    "station_id": station_id,
                    "format": "json",
                },
            )

            if response.status_code == 200:
                data = response.json()
                return self._normalize_response(data, station_id)
            else:
                logger.warning(
                    f"CPCB API returned {response.status_code} for {station_id}"
                )
                return None

        except Exception as e:
            logger.error(f"CPCB fetch failed for {station_id}: {e}")
            return None

    def _normalize_response(
        self,
        data: Dict,
        station_id: str,
    ) -> Dict:
        """
        Normalize CPCB response to match OpenAQ output format.

        Ensures both clients return identical structure for downstream processing.
        """
        pollutants = {}

        # Map CPCB field names to our internal names
        field_map = {
            "PM2.5": "pm25",
            "PM10": "pm10",
            "NO2": "no2",
            "SO2": "so2",
            "CO": "co",
            "OZONE": "o3",
            "O3": "o3",
        }

        for cpcb_name, internal_name in field_map.items():
            value = data.get(cpcb_name)
            if value is not None:
                try:
                    pollutants[internal_name] = float(value)
                except (ValueError, TypeError):
                    pass

        return {
            "station_id": station_id,
            "station_name": data.get("station", station_id),
            "latitude": data.get("latitude", 0.0),
            "longitude": data.get("longitude", 0.0),
            "timestamp": data.get("last_update",
                                  datetime.now(timezone.utc).isoformat()),
            "pollutants": pollutants,
            "source": "cpcb",
        }

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and self._http_client != "unavailable":
            await self._http_client.aclose()
            self._http_client = None
