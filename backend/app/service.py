"""
AQI Service — ORCHESTRATION LAYER

Wires together the full pipeline:
live data ingestion → physics diagnostics → ML ensemble forecast →
NLP advisory → in-memory state consumed by REST + WebSocket routes.

Runs a defensive refresh cycle so the system produces coherent output
even when upstream APIs are unreachable (demo-fallback generators).
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from backend.app.config import Settings, PROJECT_ROOT, DATA_DIR, MODELS_DIR

from backend.data.openaq_client import OpenAQClient
from backend.data.weather_client import WeatherClient
from backend.data.fire_client import FireClient
from backend.data.naqi_calculator import NAQICalculator
from backend.data.preprocessor import DataPreprocessor

from backend.physics.aisi_calculator import AISICalculator
from backend.physics.radiation_feedback import RadiationFeedback
from backend.physics.plume_transport import PlumeTransportModel
from backend.physics.emission_processor import EmissionProcessor

from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.ensemble import EnsembleForecaster

from backend.nlp.alert_generator import AlertGenerator
from backend.nlp.severity_classifier import SeverityClassifier


class AQIService:
    """
    Central service that owns data clients, physics, ML and NLP engines,
    and exposes a consistent snapshot for the API layer.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = (settings if settings is not None
                         else Settings.from_env())

        # ── Data clients ────────────────────────────────────────────
        self.openaq = OpenAQClient(
            api_key=self.settings.openaq_api_key,
            base_url=self.settings.openaq_base_url,
            rate_limit=self.settings.openaq_rate_limit,
            max_concurrency=self.settings.openaq_max_concurrency,
        )
        self.weather = WeatherClient()
        self.fire = FireClient(api_key=self.settings.nasa_firms_api_key)
        self.naqi = NAQICalculator()
        db_path = str(PROJECT_ROOT / self.settings.database_path)
        self.preprocessor = DataPreprocessor(database_path=db_path)

        # ── Physics engines ─────────────────────────────────────────
        self.aisi = AISICalculator(
            alpha=self.settings.aisi_alpha,
            beta=self.settings.aisi_beta,
            gamma=self.settings.aisi_gamma,
        )
        self.radiation = RadiationFeedback(
            latitude=self.settings.delhi_center_lat,
            longitude=self.settings.delhi_center_lon,
        )
        self.plume = PlumeTransportModel()
        self.emission_processor = EmissionProcessor()

        # ── ML stack ────────────────────────────────────────────────
        self.feature_engineer = FeatureEngineer(
            history_window=self.settings.history_window_hours,
            horizon=self.settings.forecast_horizon_hours,
        )
        self.ensemble = EnsembleForecaster(
            tft_weight=self.settings.tft_weight,
            xgb_weight=self.settings.xgb_weight,
            lgbm_weight=self.settings.lgbm_weight,
            model_dir=str(MODELS_DIR),
        )

        # ── NLP engines ─────────────────────────────────────────────
        self.severity = SeverityClassifier()
        # NLP-first: try Gemini whenever a key exists (LLM_ENHANCED=true
        # forces it even with a weak key); otherwise the local NLG engine
        # generates the advisory. Forecasting never depends on an LLM.
        llm_key = self.settings.gemini_api_key or None
        self.alert_generator = AlertGenerator(
            gemini_api_key=llm_key,
            classifier=self.severity,
        )

        # ── Domain metadata ─────────────────────────────────────────
        self.stations = self._load_stations()

        # ── Shared lock + state snapshot ────────────────────────────
        self._lock = asyncio.Lock()
        self.state: Dict[str, Any] = {
            "initialized": False,
            "last_update": None,
            "data_source": "unknown",         # live | demo
            "stations": [],                   # enriched station snapshots
            "weather": {},
            "fires": [],
            "fire_stats": {},
            "plume": {},
            "aisi": {},
            "radiation": {},
            "emissions": {},
            "forecasts": {},                  # station_id → ForecastResult
            "spatial": {},
            "alerts": [],
            "domain_summary": {},
        }

    # ────────────────────────────────────────────────────────────────
    # Public lifecycle
    # ────────────────────────────────────────────────────────────────

    async def initialize(self):
        """One-time setup: DB init and initial data refresh."""
        await self.preprocessor.initialize_database()
        await self.refresh(force=True)
        self.state["initialized"] = True

    async def refresh(self, force: bool = False) -> Dict[str, Any]:
        """
        Execute one full pipeline refresh (data → physics → ML → NLP).

        Args:
            force: Bypass the polling interval gate.

        Returns:
            A summary dict describing what was refreshed.
        """
        if not force:
            last = self.state.get("last_update")
            if last is not None:
                delta = (datetime.now(timezone.utc) -
                         datetime.fromisoformat(last)).total_seconds()
                if delta < self.settings.data_refresh_interval:
                    return {"refreshed": False, "last_update": last}

        async with self._lock:
            try:
                await self._collect_raw_data()
                await self._compute_physics()
                await self._build_station_snapshots()
                await self._compute_forecasts()
                await self._generate_alerts()
                self._build_domain_summary()

                self.state["last_update"] = datetime.now(timezone.utc).isoformat()
                return {"refreshed": True, "last_update": self.state["last_update"]}

            except Exception as e:
                logger.exception("Refresh cycle failed: %s", e)
                return {"refreshed": False, "error": str(e)}

    # ────────────────────────────────────────────────────────────────
    # Pipeline steps
    # ────────────────────────────────────────────────────────────────

    async def _collect_raw_data(self):
        """Fetch readings, weather, fires, and persist to SQLite."""
        readings = await self._safe(self.openaq.get_latest_measurements())
        mapped = self._match_readings(readings or [])

        # Fall back to demo generation when no stations resolved
        if not mapped:
            mapped = self._demo_readings()
            self.state["data_source"] = "demo"
        else:
            self.state["data_source"] = "live"

        # Normalize each station entry into persisted reading format
        records = []
        for station in self.stations:
            reading = mapped.get(station["id"])
            if reading is None:
                continue
            pollutants = reading.get("pollutants", {})
            result = self.naqi.calculate_naqi(pollutants)
            record = {
                "station_id": station["id"],
                "station_name": station["name"],
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "timestamp": reading.get("timestamp",
                                          datetime.now(timezone.utc).isoformat()),
                "pollutants": pollutants,
                "aqi": result.overall_aqi,
                "category": result.category,
                "dominant_pollutant": result.dominant_pollutant,
                "source": reading.get("source", self.state["data_source"]),
            }
            records.append(record)

        await self._safe(self.preprocessor.store_readings(records))
        self.state["raw_records"] = records

        # Weather (Delhi center)
        weather = await self._safe(self.weather.get_current_weather(
            self.settings.delhi_center_lat,
            self.settings.delhi_center_lon,
        ))
        self.state["weather"] = weather or {}

        # Fires
        fires = await self._safe(self.fire.get_active_fires(days_back=2))
        self.state["fires"] = fires or []
        self.state["fire_stats"] = self.fire.aggregate_fire_stats(
            self.state["fires"]
        )

    async def _compute_physics(self):
        """Run AISI, radiation, plume and emission engines on current data."""
        weather = dict(self.state.get("weather", {}))

        # Inject the real ERA5 PBL height for the current hour BEFORE the
        # AISI/PBL computation — otherwise the physics falls back to a
        # coarse 350/1200 m day-night heuristic that swings AISI wildly.
        wind_forecast = await self._safe(self.weather.get_hourly_forecast(
            self.settings.delhi_center_lat,
            self.settings.delhi_center_lon,
            forecast_days=2,
        ))
        if wind_forecast:
            pbl_series = wind_forecast.get("pbl_height_m") or []
            if pbl_series:
                try:
                    from zoneinfo import ZoneInfo
                    from datetime import datetime, timezone as _tz
                    now_ist = datetime.now(_tz.utc).astimezone(
                        ZoneInfo("Asia/Kolkata"))
                    fc_times = wind_forecast.get("timestamps") or []
                    idx = 0
                    for i, ts in enumerate(fc_times):
                        if str(ts)[:13] <= now_ist.strftime("%Y-%m-%dT%H"):
                            idx = i
                    pbl_now = pbl_series[min(idx, len(pbl_series) - 1)]
                    if pbl_now:
                        weather["pbl_height_m"] = float(pbl_now)
                except Exception as e:
                    logger.debug("PBL injection failed: %s", e)

        aisi_result = await self._safe(self.aisi.compute(weather))
        self.state["aisi"] = aisi_result or {}

        # Mean PM2.5 across resolved stations
        records = self.state.get("raw_records", [])
        pm25s = [r.get("pollutants", {}).get("pm25") for r in records]
        pm25s = [p for p in pm25s if p is not None]
        mean_pm25 = sum(pm25s) / len(pm25s) if pm25s else 120.0

        pblh = (aisi_result or {}).get("pbl", {}).get("pbl_height_m", 700.0)
        rad_result = await self._safe(self.radiation.compute(
            pm25=mean_pm25,
            cloud_cover_pct=weather.get("cloud_cover_pct"),
            baseline_pbl_m=pblh,
        ))
        self.state["radiation"] = rad_result or {}

        # Wind forecast for plume transport (reuses the hourly fetch above)
        plume_result = await self._safe(self.plume.compute_trajectories(
            self.state.get("fires", []),
            wind_forecast=wind_forecast,
        ))
        self.state["plume"] = plume_result or {}

        emissions = await self._safe(self.emission_processor.compute(
            fires=self.state.get("fires", []),
        ))
        self.state["emissions"] = emissions or {}

    async def _build_station_snapshots(self):
        """Attach current reading + AQI metadata to each station."""
        records = {
            r["station_id"]: r for r in self.state.get("raw_records", [])
        }
        snapshots = []

        for station in self.stations:
            record = records.get(station["id"])
            current = None
            if record:
                current = self._station_payload(record)

            # History for forecast feature building
            history = await self._safe(self.preprocessor.get_station_history(
                station["id"], hours=72
            ))

            snapshot = {
                "id": station["id"],
                "name": station["name"],
                "short_name": station["short_name"],
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "city": station["city"],
                "zone": station["zone"],
                "type": station["type"],
                "elevation_m": station["elevation_m"],
                "current": current,
                "history_count": len(history),
                "history": history[-72:],
            }
            snapshots.append(snapshot)

        self.state["stations"] = snapshots

    async def _compute_forecasts(self):
        """Generate 72-hour ensemble forecasts for every station."""
        aisi_state = self.state.get("aisi", {})
        plume = self.state.get("plume", {})
        pblh = (aisi_state.get("pbl") or {}).get("pbl_height_m", 700.0)

        # Fire contribution estimate (sum of plume contributions arriving)
        fire_contrib = sum(
            e.get("estimated_contribution_pm25", 0.0)
            for e in plume.get("arrival_estimates", [])
        )

        contexts = []
        for station in self.state.get("stations", []):
            current = station.get("current") or {}
            pollutants = current.get("pollutants", {})
            pm25 = pollutants.get("pm25")
            if pm25 is None:
                continue

            history = [r.get("pm25") for r in station.get("history", [])]
            history = [h for h in history if h is not None]

            # Per-station PM10/PM2.5 ratio from live observations (clamped),
            # else median of history, else Delhi-typical 1.35 fallback.
            pm10_ratio = self._station_pm10_ratio(
                station, pollutants.get("pm25"), pollutants.get("pm10"))

            features = self.feature_engineer.build_features(
                station_id=station["id"],
                readings=station.get("history", []),
                weather=self.state.get("weather", {}),
                fire_summary=self.state.get("fire_stats", {}),
                physics=aisi_state,
            )

            ctx = self.ensemble.build_context(
                station_id=station["id"],
                current_pm25=pm25,
                current_pm10=pollutants.get("pm10"),
                history_pm25=history,
                aisi=aisi_state.get("aisi", 2.0),
                pbl_height_m=pblh,
                inversion_strength=(aisi_state.get("pbl") or {})
                    .get("inversion_strength_k", 0.0),
                fire_contribution=fire_contrib,
                features=features,
                pm10_ratio=pm10_ratio,
            )
            contexts.append(ctx)

        forecasts = await self.ensemble.forecast_domain(contexts)
        self.state["forecasts"] = {
            f.station_id: f.to_dict() for f in forecasts
        }

        # Spatial forecast grid overlay
        self.state["spatial"] = self._build_spatial_geojson(forecasts)

    async def _generate_alerts(self):
        """Generate domain advisory from the worst station."""
        forecasts = self.state.get("forecasts", {})
        if not forecasts:
            return

        worst_station_id = max(
            forecasts,
            key=lambda sid: forecasts[sid]["aqi"][0] if forecasts[sid]["aqi"] else 0,
        )
        forecast = forecasts[worst_station_id]
        station = self._station_by_id(worst_station_id)

        aisi_state = self.state.get("aisi", {})
        fires = self.state.get("fires", [])
        plume = self.state.get("plume", {})

        trend_label, _ = self.severity.forecast_trend(forecast.get("aqi", []))
        dominants = (station or {}).get("current", {}).get("pollutants", {})
        if not dominants and self.state.get("raw_records"):
            dominants = self.state["raw_records"][0].get("pollutants", {})

        # Recompute GRAP with peak PM2.5 so stage agrees with the AQI
        # category (AISI-only GRAP caused Moderate+Stage-II mismatches).
        from backend.formulas.aisi_formulas import grap_activation_level
        peak_pm25 = max(forecast["pm25"]) if forecast.get("pm25") else 0.0
        peak_aqi = max(forecast["aqi"]) if forecast.get("aqi") else 0.0
        grap = grap_activation_level(
            aisi=aisi_state.get("aisi", 0.0), pm25=peak_pm25,
        )

        alert = await self.alert_generator.generate(
            category=forecast["category"][0] if forecast.get("category") else "Moderate",
            peak_pm25=peak_pm25,
            peak_aqi=peak_aqi,
            aisi=aisi_state.get("aisi", 0.0),
            dominants=dominants,
            corridor=(plume.get("corridor") or {}).get("name", "North-Westerly"),
            active_fires=len(fires),
            trend=trend_label,
            aq_series=forecast.get("aqi"),
            grap=grap,
            horizon_hours=len(forecast["timestamps"]),
            language="en",
        )
        alert["station_id"] = worst_station_id
        alert["station_name"] = (station or {}).get("name", "Delhi NCR")
        # Canonical feed is English-only; Hindi is rendered on demand and
        # never mixed into this list (avoids showing the same advice twice).
        # history() already includes this alert newest-first; don't duplicate.
        self.state["alerts"] = self.alert_generator.history(
            limit=10, language="en")

        # Keep the inputs so the same advisory can be re-rendered in
        # another language on demand (see generate_alert_in_language).
        self.state["alert_context"] = {
            "inputs": {
                "category": (forecast["category"][0]
                             if forecast.get("category") else "Moderate"),
                "peak_pm25": peak_pm25,
                "peak_aqi": peak_aqi,
                "aisi": aisi_state.get("aisi", 0.0),
                "dominants": dominants,
                "corridor": (plume.get("corridor") or {}).get(
                    "name", "North-Westerly"),
                "active_fires": len(fires),
                "trend": trend_label,
                "aq_series": forecast.get("aqi"),
                "grap": grap,
                "horizon_hours": len(forecast["timestamps"]),
            },
            "station_id": worst_station_id,
            "station_name": (station or {}).get("name", "Delhi NCR"),
        }

    async def generate_alert_in_language(
        self, language: str = "en"
    ) -> Optional[Dict]:
        """Regenerate the current domain advisory in another language."""
        ctx = self.state.get("alert_context")
        if not ctx:
            return None
        alert = await self.alert_generator.generate(
            **ctx["inputs"], language=language, record_history=False
        )
        alert["station_id"] = ctx.get("station_id")
        alert["station_name"] = ctx.get("station_name")
        return alert

    def _build_domain_summary(self):
        """Aggregate overall domain metrics for the header/ticker."""
        stations = self.state.get("stations", [])
        currents = [s["current"] for s in stations if s.get("current")]

        if not currents:
            self.state["domain_summary"] = {}
            return

        worst = max(currents, key=lambda c: c.get("aqi", 0))
        mean_aqi = sum(c.get("aqi", 0) for c in currents) / len(currents)
        category_counts = {}
        for c in currents:
            cat = c.get("category", "Unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        self.state["domain_summary"] = {
            "station_count": len(currents),
            "mean_aqi": round(mean_aqi, 1),
            "worst": {
                "station_id": worst.get("station_id"),
                "station_name": worst.get("station_name"),
                "aqi": worst.get("aqi"),
                "category": worst.get("category"),
                "color": worst.get("color"),
            },
            "category_counts": category_counts,
            "fire_count": len(self.state.get("fires", [])),
            "data_source": self.state.get("data_source"),
        }

    # ────────────────────────────────────────────────────────────────
    # Query helpers exposed to routes
    # ────────────────────────────────────────────────────────────────

    async def get_snapshot(self) -> Dict[str, Any]:
        """Return the full current state snapshot (for WebSocket push)."""
        return {
            "type": "snapshot",
            "last_update": self.state.get("last_update"),
            "data_source": self.state.get("data_source"),
            "domain_summary": self.state.get("domain_summary", {}),
            "stations": self.state.get("stations", []),
            "aisi": self.state.get("aisi", {}),
            "fire_stats": self.state.get("fire_stats", {}),
            "plume": self.state.get("plume", {}),
            "alerts": self.state.get("alerts", []),
            "forecasts": self.state.get("forecasts", {}),
            "spatial": self.state.get("spatial", {}),
            "radiation": self.state.get("radiation", {}),
        }

    def get_stations(self) -> List[Dict]:
        return self.state.get("stations", [])

    def get_station(self, station_id: str) -> Optional[Dict]:
        return self._station_by_id(station_id)

    def get_forecast(self, station_id: str) -> Optional[Dict]:
        return self.state.get("forecasts", {}).get(station_id)

    def get_aisi(self) -> Dict:
        aisi = self.state.get("aisi", {})
        return {
            **aisi,
            "history": self.aisi.history(limit=48),
        }

    def get_fires(self) -> Dict:
        return {
            "fires": self.state.get("fires", []),
            "stats": self.state.get("fire_stats", {}),
            "plume": self.state.get("plume", {}),
        }

    def get_alerts(self) -> List[Dict]:
        return self.state.get("alerts", [])

    def get_spatial(self) -> Dict:
        return self.state.get("spatial", {})

    def get_emissions(self) -> Dict:
        return self.state.get("emissions", {})

    def get_radiation(self) -> Dict:
        return self.state.get("radiation", {})

    # ────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────

    def _station_payload(self, record: Dict) -> Dict:
        pollutants = record.get("pollutants", {})
        result = self.naqi.calculate_naqi(pollutants)
        return {
            "station_id": record.get("station_id"),
            "station_name": record.get("station_name"),
            "timestamp": record.get("timestamp"),
            "pollutants": pollutants,
            "aqi": result.overall_aqi,
            "category": result.category,
            "color": result.color,
            "dominant_pollutant": result.dominant_pollutant,
            "sub_indices": result.sub_indices,
            "health_impact": result.health_impact,
            "source": record.get("source", self.state.get("data_source", "unknown")),
        }

    def _build_spatial_geojson(self, forecasts) -> Dict:
        """Pixel grid of interpolated PM2.5 for map heatmap overlay."""
        stations = self.state.get("stations", [])
        by_id = {f.station_id: f for f in forecasts}

        features = []
        for station in stations:
            forecast = by_id.get(station["id"])
            current = station.get("current")
            value = None
            if forecast:
                value = forecast.pm25[0]
            elif current:
                value = current.get("pollutants", {}).get("pm25")
            if value is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [station["longitude"], station["latitude"]],
                },
                "properties": {
                    "id": station["id"],
                    "name": station["short_name"],
                    "pm25": value,
                    "aqi": (forecast.aqi[0] if forecast and forecast.aqi
                            else (current or {}).get("aqi", 0)),
                    "category": (
                        (forecast.category[0] if forecast and forecast.category else None)
                        or (current or {}).get("category", "Unknown")
                    ),
                    "color": (
                        (forecast.colors[0] if forecast and forecast.colors else None)
                        or (current or {}).get("color", "#808080")
                    ),
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    @staticmethod
    def _station_name_keys(station: Dict) -> List[str]:
        """Normalized name tokens used to anchor OpenAQ matches by name."""
        import re
        keys = [station.get("short_name", ""), station.get("name", "")]
        out = []
        for k in keys:
            k = re.sub(r"[^a-z0-9 ]", " ", str(k).lower())
            k = re.sub(r"\b(delhi|new|sector|sec|phase|gram|nagar|marg|road|rd|station|dpcc|cpcb|uppcb|hspcb|imd|iitm|sai|teri)\b", " ", k)
            for tok in k.split():
                if len(tok) >= 4:
                    out.append(tok)
        return out

    def _match_readings(self, readings: List[Any]) -> Dict[str, Dict]:
        """Map OpenAQ readings to station metadata.

        Name-anchored first (an OpenAQ location whose name shares a token
        with our station wins regardless of distance), nearest-proximity
        fallback within 25 km. Fixes identical readings copied across
        neighbouring stations (e.g. ITO vs Mandir Marg).
        """
        if not readings:
            return {}

        mapped = {}
        for station in self.stations:
            lat, lon = station["latitude"], station["longitude"]
            name_keys = self._station_name_keys(station)

            # OpenAQ returns one reading per pollutant; pick the nearest
            # reading for each pollutant individually so a mid-city monitor
            # does not masquerade for a far suburb station.
            best_pollutants: Dict[str, float] = {}
            best_pollutant_dist: Dict[str, float] = {}
            best = None
            best_eff = 25.0  # km radius on effective (anchor-weighted) distance
            best_dist = 25.0

            for reading in readings:
                rlat = getattr(reading, "latitude", None)
                rlon = getattr(reading, "longitude", None)
                if rlat is None or rlon is None:
                    continue
                d = self._haversine(lat, lon, rlat, rlon)
                # Name anchor: shared token halves effective distance so the
                # correctly-named monitor wins over a nearer wrong one.
                rname = str(getattr(reading, "station_name", "") or "").lower()
                anchored = any(k in rname for k in name_keys)
                eff_d = d * 0.5 if anchored else d
                if eff_d <= 25.0:  # km radius on effective distance
                    if best is None or eff_d < best_eff:
                        best_eff = eff_d
                        best_dist = d
                        best = reading
                    for k, v in (reading.pollutants or {}).items():
                        if v is None:
                            continue
                        if k not in best_pollutant_dist or d < best_pollutant_dist[k]:
                            best_pollutant_dist[k] = d
                            best_pollutants[k] = v

            if best is not None:
                mapped[station["id"]] = {
                    "pollutants": best_pollutants,
                    "timestamp": best.timestamp,
                    "source": "live",
                    "matched_distance_km": round(best_dist, 1),
                }
        return mapped

    def _demo_readings(self) -> Dict[str, Dict]:
        """
        Generate realistic demo readings for the Delhi NCR domain.

        Mean-reverting random walk per station: values evolve smoothly
        (±3% per cycle) around a diurnal target, so consecutive snapshots
        look like a real atmosphere instead of white noise. (The old
        version multiplied the base by up to ~9x, spraying 15–400 µg/m³
        garbage that poisoned history, features and AISI.)
        """
        import random
        try:
            from zoneinfo import ZoneInfo
            hour = datetime.now(timezone.utc).astimezone(
                ZoneInfo("Asia/Kolkata")).hour
        except Exception:
            hour = datetime.now(timezone.utc).hour
        import math
        # Delhi diurnal: calm-night peak, afternoon minimum
        diurnal = 1.0 + 0.35 * math.cos((hour - 1) * math.pi / 12.0)
        month = datetime.now().month
        winter = month in (11, 12, 1, 2)
        base = 140 if winter else 55

        if not hasattr(self, "_demo_state"):
            self._demo_state = {}
        mapped = {}
        for station in self.stations:
            zone_factor = self._zone_factor(station.get("zone", ""), station["latitude"])
            target = base * zone_factor * diurnal
            prev = self._demo_state.get(station["id"], target)
            # Mean reversion (30% toward target) + small innovation
            pm25 = prev + 0.30 * (target - prev) + prev * random.uniform(-0.03, 0.03)
            pm25 = max(8.0, min(500.0, pm25))
            self._demo_state[station["id"]] = pm25
            ratio = self._demo_state.get(station["id"] + ":ratio",
                                         random.uniform(1.6, 2.2))
            self._demo_state[station["id"] + ":ratio"] = ratio
            pm10 = pm25 * ratio
            no2 = max(8.0, pm25 * 0.32)
            so2 = max(4.0, pm25 * 0.09)
            o3 = 25.0 + 30.0 * (1.0 - (diurnal - 0.65) / 0.7)
            co = max(0.3, pm25 * 0.028)

            pollutants = {"pm25": round(pm25, 1), "pm10": round(pm10, 1),
                          "no2": round(no2, 1), "so2": round(so2, 1),
                          "o3": round(max(o3, 5.0), 1), "co": round(co, 2)}
            mapped[station["id"]] = {
                "pollutants": pollutants,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "demo",
            }
        return mapped

    def _zone_factor(self, zone: str, lat: float) -> float:
        """Spatial weighting to reproduce the polluted NW corridor."""
        import re
        north = 1.0 if lat > 28.65 else 0.85
        corridors = ["North", "North West", "East"]
        if any(word in zone for word in corridors):
            return 1.12 * north
        return 0.92 * north

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        import math
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _station_pm10_ratio(station: Dict, pm25: Optional[float],
                            pm10: Optional[float]) -> float:
        """Observed PM10/PM2.5 ratio for a station, clamped to [1.0, 2.5]."""
        if pm25 and pm10 and pm25 > 5:
            return max(1.0, min(2.5, pm10 / pm25))
        ratios = []
        for r in station.get("history", []) or []:
            p25, p10 = r.get("pm25"), r.get("pm10")
            if p25 and p10 and p25 > 5:
                ratios.append(max(1.0, min(2.5, p10 / p25)))
        if ratios:
            ratios.sort()
            return ratios[len(ratios) // 2]
        return 1.35

    def _station_by_id(self, station_id: str) -> Optional[Dict]:
        for s in self.state.get("stations", []):
            if s["id"] == station_id:
                return s
        return None

    def _load_stations(self) -> List[Dict]:
        """Load Delhi NCR station metadata from data/stations.json."""
        path = DATA_DIR / "stations.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("stations", [])
        except Exception as e:
            logger.error("Failed to load stations.json: %s", e)
            return []

    async def _safe(self, coroutine, default=None):
        """Await a coroutine, returning default on any failure."""
        try:
            return await coroutine
        except Exception as e:
            logger.warning("Step failed (%s): %s",
                           coroutine.__qualname__ if hasattr(coroutine, "__qualname__") else "?",
                           e)
            return default

    async def close(self):
        """Close all HTTP clients."""
        await self._safe(self.openaq.close())
        await self._safe(self.weather.close())
        await self._safe(self.fire.close())