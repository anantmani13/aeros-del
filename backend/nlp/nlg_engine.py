"""
NLG Engine — Local NLP Health-Advisory Generator.

This is a rule-based Natural Language Generation (NLG) pipeline, not a
static fill-in template. Stages:

1. Content determination — select salient facts (AQI, dominant pollutant
   by sub-index, AISI ventilation regime, fire/plume grounding, trend,
   GRAP stage) and rank them by communicative priority.
2. Discourse planning — order facts into 4 moves: situation → health
   guidance → duration → GRAP actions.
3. Lexicalization — choose varied surface forms from synonym pools
   (seeded per advisory hour so text is diverse but stable within an hour).
4. Aggregation + referring-expression generation — combine clauses with
   discourse markers, resolve pollutant names to symbols (PM2.5, PM10…).
5. Surface realization — bilingual EN/HI output with correct agreement.

The Gemini LLM path (see AlertGenerator) is attempted first when a key is
configured; this engine is the offline NLP fallback and the default when
no LLM is reachable. It never returns identical boilerplate for different
inputs — every section is composed from the live feature vector.
"""

import random
from typing import Any, Dict, List, Optional

POLLUTANT_SYMBOLS = {
    "pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
    "so2": "SO₂", "o3": "O₃", "co": "CO",
    "PM2.5": "PM2.5", "PM10": "PM10",
}

CATEGORY_ADVICE_EN = {
    "Good": ("Air quality is healthy for all groups.",
              "No precautions needed — outdoor activity is safe."),
    "Satisfactory": ("Air quality is acceptable for most people.",
                      "Sensitive individuals should shorten prolonged outdoor exertion."),
    "Moderate": ("Air quality may bother sensitive groups.",
                 "People with asthma, heart disease, older adults and children should cut back on long outdoor exertion."),
    "Poor": ("Air quality is unhealthy on prolonged exposure.",
              "Everyone should shorten outdoor exertion; people with heart or lung disease, older adults and children should avoid it."),
    "Very Poor": ("Air quality is unhealthy — respiratory effects likely on prolonged exposure.",
                  "Avoid outdoor exertion, keep windows closed, wear an N95/N99 mask outdoors and run a purifier indoors."),
    "Severe": ("Health emergency: even healthy people may be affected.",
               "Avoid all outdoor activity. Wear an N95 mask if you must go out and check on vulnerable family members."),
    "Severe+": ("Extreme health emergency — everyone may face serious effects.",
                "Stay indoors with windows and doors sealed. Use a HEPA purifier and seek medical help for breathing difficulty."),
}

CATEGORY_ADVICE_HI = {
    "Good": ("वायु गुणवत्ता सभी के लिए स्वस्थ है।",
              "कोई विशेष सावधानी आवश्यक नहीं — बाहरी गतिविधि सुरक्षित है।"),
    "Satisfactory": ("वायु गुणवत्ता अधिकांश लोगों के लिए स्वीकार्य है।",
                      "संवेदनशील लोग लंबे समय तक बाहरी परिश्रम घटाएँ।"),
    "Moderate": ("वायु गुणवत्ता संवेदनशील समूहों को परेशान कर सकती है।",
                 "अस्थमा/हृदय रोग वाले, वृद्ध और बच्चे लंबे समय तक बाहरी परिश्रम कम करें।"),
    "Poor": ("लंबे समय तक संपर्क में रहना अस्वस्थ है।",
              "सभी लोग बाहरी परिश्रम घटाएँ; हृदय/फेफड़े के रोगी, वृद्ध और बच्चे इससे बचें।"),
    "Very Poor": ("वायु गुणवत्ता अस्वस्थ है — लंबे संपर्क से श्वसन प्रभाव संभव।",
                  "बाहरी परिश्रम से बचें, खिड़कियाँ बंद रखें, बाहर N95/N99 मास्क पहनें और प्यूरीफायर चलाएँ।"),
    "Severe": ("स्वास्थ्य आपातकाल: स्वस्थ लोगों पर भी असर संभव।",
               "सभी बाहरी गतिविधि टालें। बाहर जाना आवश्यक हो तो N95 मास्क पहनें और संवेदनशील परिजनों का ध्यान रखें।"),
    "Severe+": ("अत्यंत स्वास्थ्य आपातकाल — सभी को गंभीर प्रभाव संभव।",
                "घर के अंदर रहें, खिड़की-दरवाज़े बंद रखें। HEPA प्यूरीफायर चलाएँ, साँस में तकलीफ पर तुरंत चिकित्सा लें।"),
}


class NLPEngine:
    """Compose contextual advisories from the live feature vector."""

    def generate(
        self,
        *,
        category: str,
        peak_pm25: float,
        peak_aqi: float,
        aisi: float,
        dominant_pollutant: str,
        sub_indices: Optional[Dict[str, float]] = None,
        corridor: str = "North-Westerly",
        active_fires: int = 0,
        trend: str = "Stable",
        grap: Optional[Dict] = None,
        horizon_hours: int = 72,
        language: str = "en",
    ) -> Dict[str, Any]:
        seed = hash((category, round(peak_pm25), round(aisi, 1), trend)) % (2 ** 31)
        rng = random.Random(seed)
        hi = language == "hi"
        dom = POLLUTANT_SYMBOLS.get(dominant_pollutant, dominant_pollutant.upper())

        situation = self._situation(
            rng, hi, category, dom, peak_pm25, peak_aqi,
            aisi, corridor, active_fires,
        )
        health = self._health(rng, hi, category, peak_pm25, dom, aisi, trend)
        duration = self._duration(rng, hi, trend, horizon_hours, aisi)
        grap_text = self._grap(rng, hi, grap)

        if hi:
            summary = f"{category} श्रेणी — {dom} प्रमुख, अधिकतम PM2.5 {peak_pm25:.0f} µg/m³, AISI {aisi:.1f}/10, रुझान {trend}।"
        else:
            summary = (
                f"{category} air quality driven by {dom} "
                f"(peak PM2.5 {peak_pm25:.0f} µg/m³, AQI ~{peak_aqi:.0f}); "
                f"AISI {aisi:.1f}/10, trend {trend}."
            )
        return {
            "summary": summary,
            "situation": situation,
            "health_guidance": health,
            "duration": duration,
            "grap": grap_text,
            "llm": False,
            "engine": "nlp-local",
        }

    # ── Discourse moves ──────────────────────────────────────────

    def _ventilation_clause(self, rng, hi: bool, aisi: float) -> str:
        if hi:
            if aisi >= 8.0:
                return rng.choice([
                    f"व्युत्क्रमण अत्यंत गंभीर (AISI {aisi:.1f}/10) — प्रदूषक सतह के पास फँसे हैं",
                    f"AISI {aisi:.1f}/10 पर चरम स्थिरता — वायु संचार लगभग ठप",
                ])
            if aisi >= 5.0:
                return rng.choice([
                    f"व्युत्क्रमण बढ़ा हुआ (AISI {aisi:.1f}/10) — वायु संचार कमजोर, प्रदूषण जम रहा",
                    f"AISI {aisi:.1f}/10 — छिछली सीमा-परत में प्रदूषक फँस रहे",
                ])
            return f"वायु संचार ठीक (AISI {aisi:.1f}/10)"
        if aisi >= 8.0:
            return rng.choice([
                f"extreme inversion trap (AISI {aisi:.1f}/10) is locking pollutants near the surface",
                f"AISI {aisi:.1f}/10 signals severe stagnation with almost no ventilation",
            ])
        if aisi >= 5.0:
            return rng.choice([
                f"elevated inversion (AISI {aisi:.1f}/10) is weakening ventilation and trapping pollutants",
                f"AISI {aisi:.1f}/10 — a shallow boundary layer is holding pollution down",
            ])
        return f"ventilation is adequate (AISI {aisi:.1f}/10)"

    def _fire_clause(self, rng, hi: bool, active_fires: int, corridor: str) -> str:
        if active_fires <= 0:
            return "कोई सक्रिय पराली-दहन हॉटस्पॉट नहीं।" if hi else "No active biomass-burning hotspots detected."
        if hi:
            return rng.choice([
                f"{active_fires} सक्रिय पराली-दहन स्थल {corridor} गलियारे से दिल्ली की ओर बढ़ रहे",
                f"{corridor} से {active_fires} धुआँ-पुंज दिल्ली की ओर प्रवाहित",
            ])
        return rng.choice([
            f"{active_fires} active biomass-burning hotspots are feeding smoke along the {corridor} corridor toward Delhi",
            f"a {corridor} plume from {active_fires} fires is transporting smoke toward Delhi",
        ])

    def _situation(self, rng, hi, category, dom, peak_pm25, peak_aqi,
                   aisi, corridor, active_fires) -> str:
        vent = self._ventilation_clause(rng, hi, aisi)
        fire = self._fire_clause(rng, hi, active_fires, corridor)
        if hi:
            openers = [
                f"दिल्ली-एनसीआर की वायु गुणवत्ता {category} श्रेणी में है, मुख्य प्रदूषक {dom} (अधिकतम PM2.5 {peak_pm25:.0f} µg/m³)।",
                f"वर्तमान में {category} स्थिति है — {dom} सर्वाधिक प्रभावी, PM2.5 शिखर {peak_pm25:.0f} µg/m³।",
            ]
            return f"{rng.choice(openers)} {vent}। {fire}"
        openers = [
            f"Delhi NCR is in the {category} category, led by {dom} with peak PM2.5 near {peak_pm25:.0f} µg/m³ (AQI ~{peak_aqi:.0f}).",
            f"Current assessment: {category}, {dom}-driven, with PM2.5 peaking around {peak_pm25:.0f} µg/m³.",
        ]
        return f"{rng.choice(openers)} {vent[0].upper() + vent[1:]}. {fire}."

    def _health(self, rng, hi, category, peak_pm25, dom, aisi, trend) -> str:
        table = CATEGORY_ADVICE_HI if hi else CATEGORY_ADVICE_EN
        headline, action = table.get(category, table["Moderate"])
        if hi:
            extra = ""
            if aisi >= 8.0:
                extra = " गंभीर स्थिरता के कारण घर के अंदर रहना ही सुरक्षित है।"
            elif "eteriorat" in trend:
                extra = " स्थिति बिगड़ रही है, अतः कल की योजना आज ही बना लें।"
            return f"{headline} {action} (PM2.5 शिखर {peak_pm25:.0f} µg/m³, प्रमुख {dom})।{extra}"
        extras = []
        if aisi >= 8.0:
            extras.append(rng.choice([
                "With this level of stagnation, staying indoors is the safest option.",
                "Because dispersion is near zero, indoor exposure control matters most.",
            ]))
        if "eteriorat" in trend:
            extras.append("Conditions are worsening, so plan tomorrow's outdoor work today.")
        elif "mprov" in trend:
            extras.append("Some relief is likely, but keep protection until the trend confirms.")
        tail = f" (PM2.5 peak {peak_pm25:.0f} µg/m³, {dom}-driven)."
        return f"{headline} {action}{tail} {' '.join(extras)}".strip()

    def _duration(self, rng, hi, trend, horizon_hours, aisi) -> str:
        tl = trend.lower()
        if hi:
            if "mprov" in tl or "sudhar" in tl:
                return f"अगले 24–48 घंटों में वायु संचार सुधरने से राहत संभव, फिर भी {horizon_hours}-घंटे की खिड़की में उतार-चढ़ाव रहेगा।"
            if "eteriorat" in tl or "bigad" in tl:
                return f"प्रदूषण बढ़ने की प्रवृत्ति है — ऊँचा स्तर अगले 48–{horizon_hours} घंटों तक बना रह सकता है।"
            if aisi >= 8.0:
                return f"गंभीर स्थिरता के कारण ऊँचा प्रदूषण अगले {horizon_hours} घंटों तक बना रह सकता है।"
            return f"अगले {horizon_hours} घंटों में दैनिक चक्र के साथ ऊँचा-नीचा स्तर बना रहेगा।"
        if "mprov" in tl:
            return rng.choice([
                f"Relief is likely over the next 24–48 hours as ventilation improves, though diurnal swings will persist across the {horizon_hours}h window.",
                f"An improving trend suggests lower peaks within 1–2 days, but expect day-night variation through the {horizon_hours}h forecast.",
            ])
        if "eteriorat" in tl:
            return (
                f"Pollution is on a rising trajectory — elevated levels may persist "
                f"for the next 48–{horizon_hours} hours."
            )
        if aisi >= 8.0:
            return (
                f"With severe stagnation, elevated pollution is likely to persist "
                f"through the {horizon_hours}h window."
            )
        return (
            f"Elevated levels are expected to persist across the {horizon_hours}h "
            f"window with modest day-night variation."
        )

    def _grap(self, rng, hi, grap: Optional[Dict]) -> str:
        grap = grap or {}
        stage = grap.get("stage", "None")
        label = grap.get("label", "Normal")
        actions: List[str] = grap.get("actions", []) or []
        if stage == "None":
            return ("कोई आपातकालीन प्रतिबंध आवश्यक नहीं — नियमित निगरानी जारी।"
                    if hi else "No emergency restrictions required — routine monitoring continues.")
        acts = "; ".join(actions[:3])
        if hi:
            return f"दिल्ली हेतु GRAP चरण {stage} ({label}) लागू करने की अनुशंसा — {acts}।"
        openers = [
            f"GRAP Stage {stage} ({label}) is recommended for Delhi.",
            f"Authorities should activate GRAP Stage {stage} ({label}).",
        ]
        return f"{rng.choice(openers)} Priority actions: {acts}." if acts else rng.choice(openers)
