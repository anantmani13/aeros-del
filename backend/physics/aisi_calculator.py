"""
AISI Calculator Service — Atmospheric Inversion Severity Index

Wraps the AISI equations into a service that combines PBL diagnostics
with real-time meteorology and exposes:
- Current AISI value on a 0-10 scale
- Severity category, color, and human-readable description
- GRAP (Graded Response Action Plan) activation recommendation
- 24-hour trend for the dashboard gauge
- Threshold breach detection (AISI > 8.0 triggers extreme warning)
"""

import asyncio
import logging
from typing import Dict, List, Optional

from backend.formulas.aisi_formulas import (
    calculate_aisi,
    calculate_aisi_detailed,
    aisi_severity_category,
    grap_activation_level,
)
from backend.physics.pbl_model import PBLModel

logger = logging.getLogger(__name__)

AISI_EXTREME_THRESHOLD = 8.0


class AISICalculator:
    """
    Real-time AISI computation service.

    Combines surface meteorology and PBL diagnostics to compute the
    Atmospheric Inversion Severity Index calibrated for Delhi winters.
    """

    # EMA smoothing: suppresses single-cycle sensor spikes while tracking
    # genuine trends within ~2 cycles. MAX_STEP caps unphysical jumps.
    SMOOTHING_ALPHA = 0.65
    MAX_STEP_PER_CYCLE = 2.5

    def __init__(self, alpha: Optional[float] = None,
                 beta: Optional[float] = None,
                 gamma: Optional[float] = None):
        self.alpha = alpha if alpha is not None else 2.5
        self.beta = beta if beta is not None else 150.0
        self.gamma = gamma if gamma is not None else 3.0
        self.pbl_model = PBLModel()
        self._history: List[Dict] = []
        self._last_aisi: Optional[float] = None

    async def compute(
        self,
        weather: Dict,
        hour_ist: Optional[int] = None,
        pm25: Optional[float] = None,
    ) -> Dict:
        """
        Compute the current AISI from a weather snapshot.

        Args:
            weather: Weather dict (see WeatherClient / PBLModel).
            hour_ist: Local hour; defaults to current IST hour.
            pm25: Optional mean PM2.5 (µg/m³) so the GRAP stage agrees
                with the AQI category. If omitted, GRAP is AISI-only
                (legacy behaviour, may mismatch Severe AQI + low AISI).

        Returns:
            Dict with aisi, category, description, color, sub_terms,
            pbl diagnostics, GRAP recommendation, and trend.
        """
        pbl = await self.pbl_model.compute(weather, hour_ist=hour_ist)

        # Gradient expressed as ΔT/100m (K per 100m) — matches the AISI
        # calibration envelope (α ≈ 2.5 designed for K/100m of surface layer).
        temp_grad = pbl.get("inversion_strength_k", 0.0)

        detail = calculate_aisi_detailed(
            temp_gradient=temp_grad,
            pbl_height_m=pbl.get("pbl_height_m", 700.0),
            ri_bulk=pbl.get("ri_bulk", 0.1),
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        raw_aisi = detail["aisi"]

        # Temporal smoothing: EMA + per-cycle step cap. Kills single-cycle
        # jumps from noisy wind/RH readings without masking real trends.
        aisi = raw_aisi
        if self._last_aisi is not None:
            ema = (self.SMOOTHING_ALPHA * raw_aisi
                   + (1.0 - self.SMOOTHING_ALPHA) * self._last_aisi)
            lo = self._last_aisi - self.MAX_STEP_PER_CYCLE
            hi = self._last_aisi + self.MAX_STEP_PER_CYCLE
            aisi = max(lo, min(hi, ema))
        self._last_aisi = aisi

        category, description, color = aisi_severity_category(aisi)

        # Daytime context note so "AISI 0 at 10 AM" doesn't look broken:
        # deep PBL + lapse = well-mixed midday, inversion rebuilds at night.
        pblh = pbl.get("pbl_height_m", 700.0) or 700.0
        if aisi < 2.0 and pblh > 600 and temp_grad < 0:
            context_note = (f"Daytime well-mixed (PBL ~{pblh:.0f} m, "
                            f"lapse {temp_grad:.2f} K/100m) — "
                            f"inversion rebuilds after sunset.")
        elif aisi >= 8.0:
            context_note = ("Severe nocturnal trapping — shallow PBL + "
                            "strong inversion. Expect slow dispersion.")
        elif aisi >= 5.0:
            context_note = ("Evening/night inversion building — "
                            "ventilation degrading.")
        else:
            context_note = "Transitional mixing — watch evening trend."

        result = {
            "aisi": round(aisi, 2),
            "aisi_raw": round(raw_aisi, 2),
            "category": category,
            "description": description,
            "color": color,
            "threshold_warning": aisi > AISI_EXTREME_THRESHOLD,
            "sub_terms": {
                "temp_gradient_k_per_100m": round(temp_grad, 6),
                "pbl_height_m": pbl.get("pbl_height_m"),
                "ri_bulk": pbl.get("ri_bulk"),
                # Per-term points so you can see which physics dominates:
                "term_grad": detail["term_grad"],
                "term_pbl": detail["term_pbl"],
                "term_ri": detail["term_ri"],
            },
            "pbl": pbl,
            "grap": grap_activation_level(aisi, pm25=pm25 or 0.0),
            "context_note": context_note,
            "trend": self._trend(),
        }

        self._record_history(result)
        return result

    def _trend(self) -> Dict:
        """Compute short-term AISI trend arrow from recorded history."""
        if len(self._history) < 2:
            return {"direction": "Stable", "icon": "→", "delta": 0.0}

        recent = [h["aisi"] for h in self._history[-3:]]
        delta = recent[-1] - recent[0]

        if delta > 1.5:
            direction, icon = "Rising", "⬆⬆"
        elif delta > 0.4:
            direction, icon = "Rising", "⬆"
        elif delta < -1.5:
            direction, icon = "Falling", "⬇⬇"
        elif delta < -0.4:
            direction, icon = "Falling", "⬇"
        else:
            direction, icon = "Stable", "→"

        return {"direction": direction, "icon": icon, "delta": round(delta, 2)}

    def _record_history(self, result: Dict):
        """Keep a short rolling history for trend / sparkline use."""
        from datetime import datetime, timezone

        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aisi": result["aisi"],
        })
        if len(self._history) > 144:  # keep ~1 day of hourly points
            self._history = self._history[-144:]

    def history(self, limit: int = 48) -> List[Dict]:
        """Return recent AISI history for sparkline rendering."""
        return self._history[-limit:]

    def clear_history(self):
        """Reset the internal AISI history buffer."""
        self._history = []