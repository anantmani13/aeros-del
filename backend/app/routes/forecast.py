"""
Forecast Routes — Spatial grid, forecast trigger
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["forecast"])


def _get_service(request: Request):
    return request.app.state.service


@router.get("/forecast/spatial")
async def spatial_forecast(request: Request):
    """Gridded spatial forecast as a GeoJSON FeatureCollection."""
    service = _get_service(request)
    spatial = service.get_spatial()
    return {
        "type": "FeatureCollection",
        "features": spatial.get("features", []),
        "generated_at": service.state.get("last_update"),
    }


@router.get("/aisi/current")
async def current_aisi(request: Request):
    """Current AISI value, strength, trend and GRAP recommendation."""
    service = _get_service(request)
    return service.get_aisi()


@router.get("/radiation/current")
async def current_radiation(request: Request):
    """Aerosol-radiation feedback diagnostics."""
    service = _get_service(request)
    return service.get_radiation()


@router.post("/forecast/trigger")
async def trigger_forecast(request: Request):
    """
    Manually trigger a full refresh cycle (data → physics → ML → NLP).

    Optionally passes `force=false` to respect the polling interval.
    """
    service = _get_service(request)
    force = (request.query_params.get("force", "true").lower() != "false")
    result = await service.refresh(force=force)
    if not result.get("refreshed"):
        raise HTTPException(status_code=409, detail=result)
    return result