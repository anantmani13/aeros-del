"""
Fire & Plume Tracking Routes — Active hotspots + trajectories
"""

import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/fires", tags=["fires"])


def _get_service(request: Request):
    return request.app.state.service


@router.get("/active")
async def active_fires(request: Request):
    """Active fire hotspots with plume trajectory overlays."""
    service = _get_service(request)
    data = service.get_fires()
    return {
        "fires": data.get("fires", []),
        "stats": data.get("stats", {}),
        "plume": {
            "geojson": (data.get("plume", {}) or {}).get("geojson", {}),
            "corridor": ((data.get("plume", {}) or {}).get("corridor") or {}),
            "arrival_estimates": (data.get("plume", {}) or {}).get("arrival_estimates", []),
            "stats": ((data.get("plume", {}) or {}).get("stats") or {}),
        },
        "generated_at": service.state.get("last_update"),
    }


@router.get("/stats")
async def fire_stats(request: Request):
    """Aggregated fire statistics across the Punjab-Haryana-Delhi domain."""
    service = _get_service(request)
    return service.get_fires().get("stats", {})