"""
Severity Classifier — AQI Category, Trend & Dominant Pollutant

Wraps the NAQI calculator with forecast-aware classification logic:
- Maps any AQI value to CPCB category / color / health impact
- Identifies the dominant pollutant driving each station's AQI
- Classifies 72-hour forecast trend (Improving / Stable / Deteriorating)
- Aggregates domain-wide severity (worst station overall)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from backend.data.naqi_calculator import NAQICalculator, NAQIResult

logger = logging.getLogger(__name__)


class SeverityClassifier:
    """
    AQI severity & trend classification service used by the alert engine.
    """

    def __init__(self):
        self.naqi = NAQICalculator()

    def classify(
        self,
        pollutants: Dict[str, Optional[float]],
    ) -> Dict:
        """
        Classify a single reading into full severity metadata.

        Args:
            pollutants: Dict pollutant → concentration.

        Returns:
            Dict with aqi, category, color, dominant_pollutant,
            health_impact, advisory, and sub_indices.
        """
        result = self.naqi.calculate_naqi(pollutants)
        return self._result_to_dict(result)

    def classify_forecast_series(
        self,
        pm25: List[float],
        pm10: List[float],
    ) -> List[Dict]:
        """Classify each hour of a forecast series."""
        out = []
        for p25, p10 in zip(pm25, pm10):
            out.append(self.classify({"pm25": p25, "pm10": p10}))
        return out

    def forecast_trend(
        self,
        aqi_series: List[float],
    ) -> Tuple[str, str]:
        """
        Classify the overall direction of a 72-hour AQI series.

        Returns:
            (trend_label, trend_icon)
        """
        if len(aqi_series) < 4:
            return "Stable", "→"

        early = sum(aqi_series[:6]) / 6.0
        late = sum(aqi_series[-6:]) / 6.0
        pct_change = (late - early) / max(early, 1.0)

        if pct_change <= -0.25:
            return "Rapidly Improving", "⬇⬇"
        if pct_change < -0.10:
            return "Improving", "⬇"
        if pct_change >= 0.25:
            return "Rapidly Deteriorating", "⬆⬆"
        if pct_change > 0.10:
            return "Deteriorating", "⬆"
        return "Stable", "→"

    def domain_severity(
        self,
        station_classifiers: List[Dict],
    ) -> Dict:
        """
        Aggregate severity across all stations.

        Args:
            station_classifiers: List of per-station classify() dicts.

        Returns:
            Dict with worst AQI, worst station id, category map counts.
        """
        if not station_classifiers:
            return {
                "worst_aqi": 0,
                "worst_station_id": None,
                "worst_category": "Unknown",
                "category_counts": {},
            }

        worst = max(station_classifiers, key=lambda s: s.get("aqi", 0))
        counts = {}
        for s in station_classifiers:
            cat = s.get("category", "Unknown")
            counts[cat] = counts.get(cat, 0) + 1

        return {
            "worst_aqi": worst.get("aqi", 0),
            "worst_station_id": worst.get("station_id"),
            "worst_station_name": worst.get("station_name"),
            "worst_category": worst.get("category", "Unknown"),
            "worst_color": worst.get("color", "#808080"),
            "category_counts": counts,
        }

    def _result_to_dict(self, result: NAQIResult) -> Dict:
        return {
            "aqi": result.overall_aqi,
            "category": result.category,
            "color": result.color,
            "dominant_pollutant": result.dominant_pollutant,
            "health_impact": result.health_impact,
            "advisory": result.advisory,
            "sub_indices": result.sub_indices,
        }