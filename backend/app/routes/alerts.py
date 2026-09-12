"""
Alert Routes — NLP-Generated Health Advisories
"""

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


def _get_service(request: Request):
    return request.app.state.service


@router.get("")
async def list_alerts(request: Request, limit: int = 10, lang: str = "en"):
    """List generated advisories (newest first).

    Pass `lang=hi` to re-render the current advisory in Hindi —
    same data through the NLP pipeline (LLM when reachable,
    otherwise the local NLG engine).
    """
    service = _get_service(request)
    if lang != "en":
        alert = await service.generate_alert_in_language(lang)
        if not alert:
            raise HTTPException(status_code=404, detail="No alerts generated yet")
        return {"count": 1, "alerts": [alert], "language": lang}
    alerts = service.get_alerts()[:max(limit, 1)]
    return {"count": len(alerts), "alerts": alerts, "language": "en"}


@router.get("/current")
async def current_alert(request: Request, lang: str = "en"):
    """Current NLP-generated health advisory (`?lang=hi` for Hindi)."""
    service = _get_service(request)
    if lang != "en":
        alert = await service.generate_alert_in_language(lang)
        if not alert:
            raise HTTPException(status_code=404, detail="No alerts generated yet")
        return alert
    alerts = service.get_alerts()
    if not alerts:
        raise HTTPException(status_code=404, detail="No alerts generated yet")
    return alerts[0]