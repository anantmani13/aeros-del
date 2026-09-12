"""
Alert Templates & Prompt Library — English + Hindi

Structured message templates used by the AlertGenerator to produce
human-readable, action-oriented health advisories. Each template is a
callable receiving dynamic values and returning formatted text.

Supports:
- Situation summaries (with fire/plume context)
- Health guidance per AQI severity category
- Duration estimates
- GRAP stage recommendations
- Multi-language output (en / hi)
"""

from typing import Any, Callable, Dict

# ──────────────────────────────────────────────────────────────────────
# English Templates
# ──────────────────────────────────────────────────────────────────────

def en_situation_summary(values: Dict[str, Any]) -> str:
    aqi_cat = values.get("category", "Moderate")
    dominant = values.get("dominant_pollutant", "PM2.5")
    aisi = values.get("aisi", 0.0)
    corridor = values.get("corridor", "North-Westerly")

    parts = [
        f"Air quality across Delhi NCR is in the **{aqi_cat}** category, "
        f"driven primarily by {dominant}."
    ]
    if aisi >= 8.0:
        parts.append(
            f"The Atmospheric Inversion Severity Index stands at a critical "
            f"**{aisi:.1f}/10**, indicating an extreme inversion trap that "
            f"will keep pollutants locked near the surface."
        )
    elif aisi >= 5.0:
        parts.append(
            f"Inversion severity is elevated at **{aisi:.1f}/10**, reducing "
            f"atmospheric ventilation and promoting pollutant accumulation."
        )
    else:
        parts.append(
            f"Ventilation is currently adequate (AISI {aisi:.1f}/10)."
        )

    if values.get("active_fires", 0) > 0:
        parts.append(
            f"{values['active_fires']} active biomass-burning hotspots are "
            f"being tracked across {corridor} with plume trajectories toward "
            f"Delhi."
        )

    return " ".join(parts)


def en_health_guidance(values: Dict[str, Any]) -> str:
    category = values.get("category", "Moderate")
    pm = values.get("peak_pm25", 0.0)

    guidance = {
        "Good": "Enjoy outdoor activities. Air quality poses no significant risk.",
        "Satisfactory": (
            "Acceptable air quality. Sensitive individuals should limit "
            "prolonged outdoor exertion."
        ),
        "Moderate": (
            f"People with lung or heart disease, older adults, and children "
            f"should reduce prolonged outdoor exertion (PM2.5 peaking at "
            f"{pm:.0f} µg/m³)."
        ),
        "Poor": (
            f"Everyone should reduce prolonged outdoor exertion. People with "
            f"heart or lung disease should avoid outdoor activity (PM2.5 "
            f"peaking at {pm:.0f} µg/m³)."
        ),
        "Very Poor": (
            f"Everyone should avoid outdoor exertion. Keep windows closed, "
            f"use N95/N99 masks outdoors, and run air purifiers indoors. "
            f"PM2.5 may reach {pm:.0f} µg/m³."
        ),
        "Severe": (
            f"Health emergency conditions. Avoid all outdoor activity, wear "
            f"N95 masks if unavoidable, and monitor vulnerable family members. "
            f"PM2.5 forecast up to {pm:.0f} µg/m³."
        ),
        "Severe+": (
            "EXTREME health emergency. Remain indoors with windows sealed. "
            "Use HEPA purifiers and seek medical help if breathing difficulty "
            "develops."
        ),
    }
    return guidance.get(category, guidance["Moderate"])


def en_duration_estimate(values: Dict[str, Any]) -> str:
    horizon = values.get("horizon_hours", 72)
    improving = values.get("trend", "").lower()
    if "improv" in improving:
        return (
            f"Conditions are expected to improve over the next 24–48 hours "
            f"as ventilation increases."
        )
    if "deterior" in improving:
        return (
            f"Pollution is expected to rise steadily; elevated levels may "
            f"persist for the next 48–{horizon} hours."
        )
    return (
        f"Elevated pollution levels are expected to persist over the "
        f"current {horizon}-hour forecast window, with modest diurnal "
        f"variation."
    )


def en_grap_recommendation(values: Dict[str, Any]) -> str:
    grap = values.get("grap", {})
    stage = grap.get("stage", "None")
    label = grap.get("label", "Normal")

    if stage == "None":
        return "No emergency restrictions required. Routine monitoring continues."
    return (
        f"**GRAP Stage {stage}** ({label}) recommended for Delhi. Actions "
        f"under consideration: {'; '.join(grap.get('actions', [])[:3])}."
    )


# ──────────────────────────────────────────────────────────────────────
# Hindi Templates
# ──────────────────────────────────────────────────────────────────────

def hi_situation_summary(values: Dict[str, Any]) -> str:
    aqi_cat = values.get("category", "Moderate")
    aisi = values.get("aisi", 0.0)
    corridor = values.get("corridor", "उत्तर-पश्चिम")

    parts = [
        f"दिल्ली-एनसीआर की वायु गुणवत्ता इस समय **{aqi_cat}** श्रेणी में है।"
    ]
    if aisi >= 8.0:
        parts.append(
            f"वायुमंडलीय व्युत्क्रमण सूचकांक (AISI) **{aisi:.1f}/10** पर है, "
            f"जो अत्यंत गंभीर स्थिति को दर्शाता है — प्रदूषक सतह के पास फँसे रहेंगे।"
        )
    elif aisi >= 5.0:
        parts.append(
            f"व्युत्क्रमण की तीव्रता बढ़ी हुई है (AISI {aisi:.1f}/10), जिससे वायु "
            f"संचार प्रभावित होकर प्रदूषण जमा हो रहा है।"
        )
    else:
        parts.append(f"वर्तमान में वायु संचार ठीक है (AISI {aisi:.1f}/10)।")

    if values.get("active_fires", 0) > 0:
        parts.append(
            f"{values['active_fires']} सक्रिय पराली दहन स्थल {corridor} से दिल्ली "
            f"की ओर गुजरने वाले धुएँ के प्रवाह के साथ ट्रैक किए गए हैं।"
        )
    return " ".join(parts)


def hi_health_guidance(values: Dict[str, Any]) -> str:
    category = values.get("category", "Moderate")
    pm = values.get("peak_pm25", 0.0)

    guidance = {
        "Good": "बाहरी गतिविधियों का आनंद लें। वायु गुणवत्ता से कोई खतरा नहीं।",
        "Satisfactory": (
            "वायु गुणवत्ता स्वीकार्य है। संवेदनशील लोग लंबे समय तक बाहरी परिश्रम से बचें।"
        ),
        "Moderate": (
            f"फेफड़े/हृदय रोग वाले लोग, वृद्ध और बच्चे बाहरी परिश्रम कम करें "
            f"(PM2.5 लगभग {pm:.0f} µg/m³)।"
        ),
        "Poor": (
            f"सभी लोग बाहरी परिश्रम कम करें। हृदय या फेफड़े की बीमारी वाले लोग बाहरी "
            f"गतिविधि से बचें (PM2.5 लगभग {pm:.0f} µg/m³)।"
        ),
        "Very Poor": (
            f"सभी लोग बाहरी परिश्रम से बचें। खिड़कियाँ बंद रखें, बाहर N95/N99 मास्क "
            f"पहनें और घर में एयर प्यूरीफायर चलाएँ। PM2.5 {pm:.0f} µg/m³ तक पहुँच सकता है।"
        ),
        "Severe": (
            f"स्वास्थ्य आपातकाल जैसी स्थिति। सभी बाहरी गतिविधियाँ बंद करें, आवश्यक होने पर "
            f"N95 मास्क पहनें। PM2.5 {pm:.0f} µg/m³ तक।"
        ),
        "Severe+": (
            "अत्यंत आपातकाल। घर के अंदर रहें, खिड़कियाँ सील करें, HEPA प्यूरीफायर उपयोग करें। "
            "साँस लेने में कठिनाई पर तुरंत चिकित्सा सहायता लें।"
        ),
    }
    return guidance.get(category, guidance["Moderate"])


def hi_duration_estimate(values: Dict[str, Any]) -> str:
    improving = values.get("trend", "").lower()
    horizon = values.get("horizon_hours", 72)
    if "improv" in improving:
        return "आगामी 24–48 घंटों में स्थिति में सुधार की संभावना है।"
    if "deterior" in improving:
        return (
            f"प्रदूषण लगातार बढ़ने की संभावना है; ऊँचा स्तर अगले 48–{horizon} घंटे रह सकता है।"
        )
    return (
        f"अगले {horizon}-घंटे के पूर्वानुमान में ऊँचा प्रदूषण स्तर बना रहने की संभावना है।"
    )


def hi_grap_recommendation(values: Dict[str, Any]) -> str:
    grap = values.get("grap", {})
    stage = grap.get("stage", "None")
    label = grap.get("label", "Normal")

    if stage == "None":
        return "वर्तमान में कोई आपातकालीन प्रतिबंध आवश्यक नहीं। नियमित निगरानी जारी।"
    return (
        f"दिल्ली के लिए **GRAP स्टेज {stage}** ({label}) की अनुशंसा की जाती है। "
        f"प्रस्तावित कार्य: {'; '.join(grap.get('actions', [])[:3])}।"
    )


# ──────────────────────────────────────────────────────────────────────
# Template Registry
# ──────────────────────────────────────────────────────────────────────

TEMPLATE_REGISTRY: Dict[str, Dict[str, Callable[[Dict], str]]] = {
    "en": {
        "situation_summary": en_situation_summary,
        "health_guidance": en_health_guidance,
        "duration_estimate": en_duration_estimate,
        "grap_recommendation": en_grap_recommendation,
    },
    "hi": {
        "situation_summary": hi_situation_summary,
        "health_guidance": hi_health_guidance,
        "duration_estimate": hi_duration_estimate,
        "grap_recommendation": hi_grap_recommendation,
    },
}

# Alternative short labels (bilingual copy · fallback vocabulary)
CATEGORY_HINTS = {
    "Good":        {"en": "Good",        "hi": "अच्छी"},
    "Satisfactory": {"en": "Satisfactory", "hi": "संतोषजनक"},
    "Moderate":    {"en": "Moderate",    "hi": "मध्यम"},
    "Poor":        {"en": "Poor",        "hi": "खराब"},
    "Very Poor":   {"en": "Very Poor",   "hi": "बहुत खराब"},
    "Severe":      {"en": "Severe",      "hi": "गंभीर"},
    "Severe+":     {"en": "Severe+",     "hi": "अति गंभीर"},
}

# LLM prompt template used when a Gemini/OpenAI key is configured
LLM_SYSTEM_PROMPT = (
    "You are an air quality advisory officer for Delhi NCR. Write a concise, "
    "action-oriented health advisory (max 140 words) using the following "
    "structured data. Mention the AQI category, primary pollutant, inversion "
    "severity (AISI), biomass fire activity, and the recommended GRAP stage. "
    "Output in {language}. Data: {context}"
)