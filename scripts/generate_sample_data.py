#!/usr/bin/env python
"""
Generate Sample Forecast Data — Demo/OFFLINE Mode Helper

Writes pre-computed 72-hour forecast GeoJSON + per-station JSON into
data/sample_forecasts so the system (or documentation) can show
realistic output without live API access.

Usage:
    python scripts/generate_sample_data.py
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.naqi_calculator import NAQICalculator
from backend.ml.xgboost_forecaster import statistical_baseline


def build_forecast(station_id: str, name: str, lat: float, lon: float,
                   current_pm25: float, aisi: float) -> dict:
    """Build a sample 72h forecast dict for a station."""
    baseline = statistical_baseline(
        current_pm25=current_pm25,
        history_pm25=[],
        aisi=aisi,
        pbl_height_m=600.0,
        horizon=72,
    )
    naqi = NAQICalculator()

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps, aqi_s, cat_s, col_s = [], [], [], []

    for i, pm25 in enumerate(baseline["pm25"]):
        pm10 = baseline["pm10"][i]
        result = naqi.calculate_naqi({"pm25": pm25, "pm10": pm10})
        timestamps.append((now + timedelta(hours=i + 1)).isoformat())
        aqi_s.append(result.overall_aqi)
        cat_s.append(result.category)
        col_s.append(result.color)

    return {
        "station_id": station_id,
        "station_name": name,
        "generated_at": now.isoformat(),
        "timestamps": timestamps,
        "pm25": baseline["pm25"],
        "pm10": baseline["pm10"],
        "lower": baseline["lower"],
        "upper": baseline["upper"],
        "aqi": aqi_s,
        "category": cat_s,
        "colors": col_s,
        "models": {"tft": False, "xgboost": False, "baseline": True},
    }


def build_spatial(stations: list) -> dict:
    """Build the spatial GeoJSON overlay from station forecasts."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [s["lon"], s["lat"]]},
                "properties": {
                    "id": s["id"],
                    "name": s["short_name"],
                    "pm25": s["seed_pm25"],
                    "aqi": s["seed_aqi"],
                    "category": s["seed_category"],
                    "color": s["seed_color"],
                },
            }
            for s in stations
        ],
    }


async def main():
    out_dir = PROJECT_ROOT / "data" / "sample_forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    stations_path = PROJECT_ROOT / "data" / "stations.json"
    with open(stations_path, "r", encoding="utf-8") as f:
        stations_data = json.load(f)

    naqi = NAQICalculator()
    seeds = []
    forecast_files = []

    for s in stations_data["stations"]:
        # Winter-ish deterministic seed for sample realism
        base = 92
        lat_factor = 1.12 if s["latitude"] > 28.65 else 0.95
        pm25 = round(base * lat_factor * (s["id"].__len__() % 3 + 8) / 9, 1)
        result = naqi.calculate_naqi({"pm25": pm25, "pm10": pm25 * 1.35})

        forecast = build_forecast(s["id"], s["name"], s["latitude"],
                                  s["longitude"], pm25, aisi=6.2)
        fname = f"{s['id']}.json"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(forecast, f, indent=2)
        forecast_files.append(fname)

        seeds.append({
            "id": s["id"],
            "short_name": s["short_name"],
            "lat": s["latitude"],
            "lon": s["longitude"],
            "seed_pm25": pm25,
            "seed_aqi": result.overall_aqi,
            "seed_category": result.category,
            "seed_color": result.color,
        })

    spatial = build_spatial(seeds)
    with open(out_dir / "spatial.json", "w", encoding="utf-8") as f:
        json.dump(spatial, f, indent=2)

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Sample 72h forecasts for offline demo mode",
        "station_forecasts": forecast_files,
        "spatial_file": "spatial.json",
    }
    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"[OK] Generated {len(forecast_files)} station forecasts + spatial overlay")
    print(f"     Output: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())