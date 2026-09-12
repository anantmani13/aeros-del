"""
Alert Generator — NLP Health Advisory (LLM-first, local-NLG fallback).

Pipeline:
1. Severity grounding — dominant pollutant is resolved from NAQI
   sub-indices (not raw µg/m³), via SeverityClassifier. GRAP is
   reconciled so it can never contradict the AQI category.
2. LLM path — Gemini (new `google.genai` SDK, old `google.generativeai`
   as fallback) with a structured JSON prompt producing 4 DISTINCT
   sections: situation / health_guidance / duration / grap.
3. Local NLP path — NLPEngine NLG pipeline (content determination →
   discourse planning → lexicalization → surface realization, EN/HI).
4. Legacy templates — last-resort safety net only.

`generator` field in output: "llm-gemini" | "nlp-local" | "templates-legacy".
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.nlp.templates import TEMPLATE_REGISTRY, CATEGORY_HINTS
from backend.nlp.severity_classifier import SeverityClassifier
from backend.nlp.nlg_engine import NLPEngine

logger = logging.getLogger(__name__)

SEVERITY_META = {
    "Good": {"level": "Low", "priority": 1},
    "Satisfactory": {"level": "Low", "priority": 2},
    "Moderate": {"level": "Moderate", "priority": 3},
    "Poor": {"level": "Moderate", "priority": 4},
    "Very Poor": {"level": "High", "priority": 5},
    "Severe": {"level": "Critical", "priority": 6},
    "Severe+": {"level": "Critical", "priority": 7},
}

# CPCB GRAP mapping by AQI category (authoritative; AISI may escalate by ≤1).
CATEGORY_GRAP = {
    "Good": "None", "Satisfactory": "None", "Moderate": "None",
    "Poor": "I", "Very Poor": "II", "Severe": "III", "Severe+": "IV",
}
GRAP_ORDER = ["None", "I", "II", "III", "IV"]


def reconcile_grap(category: str, grap: Optional[Dict]) -> Dict:
    """Force GRAP stage to agree with the AQI category.

    Rule: start from the category-mapped stage; allow the physics (AISI)
    stage to escalate by at most one step, never to contradict downwards
    by more than one step. Fixes Moderate+Stage-II type mismatches.
    """
    grap = dict(grap or {})
    expected = CATEGORY_GRAP.get(category, "None")
    current = grap.get("stage", "None")
    if current not in GRAP_ORDER:
        current = "None"
    if expected not in GRAP_ORDER:
        expected = "None"
    ei, ci = GRAP_ORDER.index(expected), GRAP_ORDER.index(current)
    if ci > ei + 1:
        grap["stage"] = GRAP_ORDER[ei + 1]
    elif ci < ei:
        grap["stage"] = expected
    # Keep label consistent with the (possibly capped) stage.
    stage_labels = {"None": "Normal", "I": "Poor", "II": "Very Poor",
                    "III": "Severe", "IV": "Emergency"}
    grap["label"] = stage_labels.get(grap.get("stage"), "Normal")
    if grap.get("stage") == "None":
        grap["actions"] = ["Routine monitoring and enforcement"]
    return grap


class AlertGenerator:
    """Contextual advisory generator (LLM-first, NLP-local fallback)."""

    def __init__(self, gemini_api_key: Optional[str] = None,
                 classifier: Optional[SeverityClassifier] = None):
        self.gemini_api_key = (gemini_api_key or "").strip() or None
        self.classifier = classifier or SeverityClassifier()
        self.nlg = NLPEngine()
        self._history: List[Dict] = []

    async def generate(
        self,
        category: str,
        peak_pm25: float,
        aisi: float,
        dominants: Dict[str, Optional[float]],
        corridor: str = "North-Westerly",
        active_fires: int = 0,
        trend: str = "Stable",
        aq_series: Optional[List[float]] = None,
        grap: Optional[Dict] = None,
        horizon_hours: int = 72,
        language: str = "en",
        peak_aqi: Optional[float] = None,
        record_history: bool = True,
    ) -> Dict:
        # 1. Severity grounding via NAQI sub-indices (fixes raw-max bug).
        dominant_pollutant, sub_indices = self._resolve_dominant(dominants)
        grap = reconcile_grap(category, grap)
        if peak_aqi is None and aq_series:
            try:
                peak_aqi = max(float(x) for x in aq_series if x is not None)
            except ValueError:
                peak_aqi = 0.0
        peak_aqi = float(peak_aqi or 0.0)

        # 2. LLM path (distinct sections, structured JSON).
        llm_blocks = None
        if self.gemini_api_key:
            llm_blocks = await self._llm_advisory(
                category, peak_pm25, peak_aqi, aisi, dominant_pollutant,
                corridor, active_fires, trend, grap, horizon_hours, language,
            )
            if not llm_blocks:
                logger.info("LLM advisory unavailable — using local NLP engine")

        # 3. Local NLP engine.
        generator = "llm-gemini"
        if not llm_blocks:
            try:
                llm_blocks = self.nlg.generate(
                    category=category, peak_pm25=peak_pm25, peak_aqi=peak_aqi,
                    aisi=aisi, dominant_pollutant=dominant_pollutant,
                    sub_indices=sub_indices, corridor=corridor,
                    active_fires=active_fires, trend=trend, grap=grap,
                    horizon_hours=horizon_hours, language=language,
                )
                generator = "nlp-local"
            except Exception as e:  # pragma: no cover
                logger.warning("NLG engine failed (%s) — legacy templates", e)
                llm_blocks = self._template_advisory(
                    category, peak_pm25, aisi, dominant_pollutant,
                    corridor, active_fires, trend, grap, horizon_hours, language,
                )
                generator = "templates-legacy"

        severity = SEVERITY_META.get(category, {"level": "Moderate", "priority": 3})
        if aisi >= 8.0:
            severity = {"level": "Critical", "priority": 7}

        alert = {
            "id": str(uuid.uuid4())[:13],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "title": self._title(category, language),
            "summary": llm_blocks.get("summary"),
            "sections": {
                "situation": llm_blocks.get("situation"),
                "health_guidance": llm_blocks.get("health_guidance"),
                "duration": llm_blocks.get("duration"),
                "grap": llm_blocks.get("grap"),
            },
            "severity": severity,
            "category": category,
            "category_label": CATEGORY_HINTS.get(category, {}).get(language, category),
            "peak_pm25": round(float(peak_pm25 or 0.0), 1),
            "peak_aqi": round(peak_aqi, 1),
            "aisi": round(float(aisi or 0.0), 2),
            "dominant_pollutant": dominant_pollutant,
            "sub_indices": sub_indices,
            "active_fires": active_fires,
            "corridor": corridor,
            "trend": trend,
            "grap_stage": (grap or {}).get("stage", "None"),
            "grap": grap,
            "horizon_hours": horizon_hours,
            "generator": generator,
        }

        if record_history:
            self._history.append(alert)
            if len(self._history) > 100:
                self._history = self._history[-100:]
        return alert

    def history(self, limit: int = 20, language: Optional[str] = None) -> List[Dict]:
        """Return generated alerts, newest first, optionally one language."""
        items = self._history
        if language is not None:
            items = [a for a in items if a.get("language") == language]
        return list(reversed(items[-limit:]))

    def _resolve_dominant(self, dominants: Dict[str, Optional[float]]):
        """Pick dominant pollutant by NAQI sub-index, not raw µg/m³."""
        dominants = dominants or {}
        try:
            info = self.classifier.classify(dominants)
            dom = info.get("dominant_pollutant") or "pm25"
            return dom, info.get("sub_indices", {})
        except Exception:
            pass
        # Fallback: raw max with sane key normalization.
        if dominants:
            key = max(dominants, key=lambda k: dominants[k] or 0)
            return key, {}
        return "pm25", {}

    # ── Template path (legacy safety net) ────────────────────────────

    def _template_advisory(
        self,
        category: str,
        peak_pm25: float,
        aisi: float,
        dominant_pollutant: str,
        corridor: str,
        active_fires: int,
        trend: str,
        grap: Optional[Dict],
        horizon_hours: int,
        language: str,
    ) -> Dict:
        templates = TEMPLATE_REGISTRY.get(language, TEMPLATE_REGISTRY["en"])
        values = {
            "category": category,
            "dominant_pollutant": dominant_pollutant,
            "aisi": aisi,
            "corridor": corridor,
            "active_fires": active_fires,
            "trend": trend,
            "peak_pm25": peak_pm25,
            "horizon_hours": horizon_hours,
            "grap": grap,
        }
        return {
            "summary": templates["situation_summary"](values),
            "situation": templates["situation_summary"](values),
            "health_guidance": templates["health_guidance"](values),
            "duration": templates["duration_estimate"](values),
            "grap": templates["grap_recommendation"](values),
            "llm": False,
            "engine": "templates-legacy",
        }

    # ── LLM path (Gemini, structured sections) ───────────────────────

    def _llm_prompt(self, language: str, ctx: Dict[str, Any]) -> str:
        lang_name = "Hindi (Devanagari)" if language == "hi" else "English"
        return (
            "You are an air-quality health advisory officer for Delhi NCR. "
            f"Write the advisory in {lang_name} as strict JSON with exactly these "
            "keys: situation (2-3 sentences: AQI category, dominant pollutant "
            "with value, AISI ventilation, fire/plume context), health_guidance "
            "(2-3 sentences: who is at risk and precise protective actions), "
            "duration (1-2 sentences: how long elevated levels persist given "
            "trend), grap (1-2 sentences: GRAP stage + top actions). "
            "Ground every claim in the data; no markdown, no extra keys, "
            "max 180 words total. Data: " + json.dumps(ctx)
        )

    async def _llm_advisory(
        self,
        category: str,
        peak_pm25: float,
        peak_aqi: float,
        aisi: float,
        dominant_pollutant: str,
        corridor: str,
        active_fires: int,
        trend: str,
        grap: Optional[Dict],
        horizon_hours: int,
        language: str,
    ) -> Optional[Dict]:
        ctx = {
            "aqi_category": category,
            "peak_pm25": round(float(peak_pm25 or 0), 1),
            "peak_aqi": round(float(peak_aqi or 0), 1),
            "aisi": round(float(aisi or 0), 2),
            "dominant_pollutant": dominant_pollutant,
            "corridor": corridor,
            "active_fires": active_fires,
            "trend": trend,
            "grap_stage": (grap or {}).get("stage", "None"),
            "grap_label": (grap or {}).get("label", "Normal"),
            "horizon_hours": horizon_hours,
        }
        text = await self._gemini_complete(self._llm_prompt(language, ctx))
        if not text:
            return None
        parsed = self._parse_sections(text)
        if not parsed:
            return None
        parsed["llm"] = True
        parsed["engine"] = "llm-gemini"
        # Summary is a compact headline — never a copy of the situation body.
        from backend.nlp.nlg_engine import POLLUTANT_SYMBOLS
        dom_sym = POLLUTANT_SYMBOLS.get(dominant_pollutant,
                                        dominant_pollutant.upper())
        if language == "hi":
            parsed["summary"] = (
                f"{category} श्रेणी — {dom_sym} प्रमुख, "
                f"PM2.5 शिखर {peak_pm25:.0f}, AISI {aisi:.1f}/10, रुझान {trend}।"
            )
        else:
            parsed["summary"] = (
                f"{category} air quality, {dom_sym}-driven "
                f"(PM2.5 peak {peak_pm25:.0f}, AQI ~{peak_aqi:.0f}); "
                f"AISI {aisi:.1f}/10, {trend.lower()} trend."
            )
        return parsed

    # Model preference order (retired models fall through to the next).
    GEMINI_MODELS = ("gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash")

    async def _gemini_complete(self, prompt: str) -> Optional[str]:
        # New SDK first (google-genai), legacy SDK as fallback.
        # 25s cap so Render startup never hangs on LLM.
        try:
            from google import genai as new_genai  # type: ignore
            client = new_genai.Client(api_key=self.gemini_api_key)
            for model_name in self.GEMINI_MODELS:
                try:
                    resp = await asyncio.wait_for(asyncio.to_thread(
                        client.models.generate_content,
                        model=model_name,
                        contents=prompt,
                    ), timeout=25)
                    text = getattr(resp, "text", "") or ""
                    if text.strip():
                        return text.strip()
                except Exception as e:
                    logger.debug("model %s failed: %s", model_name, e)
        except Exception as e:
            logger.debug("new genai SDK failed: %s", e)
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.gemini_api_key)
            for model_name in self.GEMINI_MODELS:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = await asyncio.to_thread(
                        model.generate_content, prompt)
                    text = getattr(response, "text", "") or ""
                    if text.strip():
                        return text.strip()
                except Exception as e:
                    logger.debug("legacy model %s failed: %s", model_name, e)
            return None
        except Exception as e:
            logger.warning("LLM advisory failed: %s", e)
            return None

    @staticmethod
    def _parse_sections(text: str) -> Optional[Dict]:
        """Parse 4 distinct sections from JSON (or best-effort fallback)."""
        cleaned = text.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            start, end = cleaned.index("{"), cleaned.rindex("}") + 1
            data = json.loads(cleaned[start:end])
            sit = str(data.get("situation", "")).strip()
            health = str(data.get("health_guidance", "")).strip()
            dur = str(data.get("duration", "")).strip()
            grap = str(data.get("grap", "")).strip()
            if sit and health:
                # Guard against single-blob duplication.
                if len({sit, health, dur, grap}) < 2:
                    return None
                return {"situation": sit, "health_guidance": health,
                        "duration": dur or sit, "grap": grap or sit}
        except Exception:
            pass
        # Fallback: split plain text into sentences across sections.
        sents = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if len(sents) >= 4:
            j = lambda parts: ". ".join(parts).strip() + "."
            return {"situation": j(sents[0:2]), "health_guidance": j(sents[2:4]),
                    "duration": j(sents[4:5]) if len(sents) > 4 else j(sents[3:4]),
                    "grap": j(sents[5:6]) if len(sents) > 5 else j(sents[3:4])}
        return None

    # ── Helpers ──────────────────────────────────────────────────────

    def _title(self, category: str, language: str) -> str:
        if language == "hi":
            labels = {
                "Good": "वायु गुणवत्ता: अच्छी",
                "Satisfactory": "वायु गुणवत्ता: संतोषजनक",
                "Moderate": "वायु गुणवत्ता: मध्यम",
                "Poor": "वायु गुणवत्ता: खराब",
                "Very Poor": "वायु गुणवत्ता: बहुत खराब",
                "Severe": "वायु गुणवत्ता: गंभीर",
                "Severe+": "वायु गुणवत्ता: अति गंभीर",
            }
            return labels.get(category, f"वायु गुणवत्ता परामर्श: {category}")
        return f"Air Quality Advisory — {category}"
