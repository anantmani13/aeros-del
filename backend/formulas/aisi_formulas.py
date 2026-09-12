"""
AISI (Atmospheric Inversion Severity Index) Equations — WRF-Chem Derived

Implements:
- Surface temperature gradient computation
- AISI composite index (0-10 scale)
- Severity categorization
- GRAP (Graded Response Action Plan) activation levels
"""

from typing import Tuple


# ── AISI Calibration Constants (Delhi Winter) ────────────────────────
# NOTE on units: temp_gradient here is ΔT per 100 m (K/100m), NOT K/m.
# Typical Delhi winter: night inversion +1 to +4 K/100m, day lapse ≈ -1 K/100m.
# α=2.5 maps +1.5K → 3.75 pts, +3K → 7.5 pts (severe). If you pass K/m,
# multiply by 100 first (0.015 K/m = 1.5 K/100m).
DEFAULT_ALPHA = 2.5    # temperature gradient weight (per K/100m)
DEFAULT_BETA = 150.0   # inverse PBL height weight
DEFAULT_GAMMA = 3.0    # Richardson number weight
AISI_MAX = 10.0

# Clamp Ri_b so one noisy wind reading can't swing AISI by ±5 points.
RI_MIN = -0.5
RI_MAX = 2.0
# Clamp gradient so extreme synthetic values stay in calibration envelope.
GRAD_MIN = -2.0  # K/100m (strong daytime lapse)
GRAD_MAX = 5.0   # K/100m (extreme smog-night inversion)


def temperature_gradient(
    t_surface_k: float,
    t_upper_k: float,
    dz_m: float = 100.0,
    per_100m: bool = True,
) -> float:
    """
    Calculate vertical temperature gradient (∂T/∂z).

    Positive gradient = temperature increases with height = inversion.

    Args:
        t_surface_k: Surface temperature (K or °C — same unit required)
        t_upper_k: Upper-level temperature (K or °C)
        dz_m: Height difference (m), default 100m
        per_100m: If True (default) return K/100m to match AISI
            calibration. If False return K/m.

    Returns:
        Temperature gradient (K/100m by default, K/m if per_100m=False)
    """
    grad_per_m = (t_upper_k - t_surface_k) / max(dz_m, 1.0)
    return grad_per_m * 100.0 if per_100m else grad_per_m


def calculate_aisi(
    temp_gradient: float,
    pbl_height_m: float,
    ri_bulk: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """
    Calculate Atmospheric Inversion Severity Index (AISI).

    AISI = min(10.0, α*(ΔT/100m) + β*(1/max(PBLH, 50)) + γ*Ri_b_clamped)

    Designed so that:
    - AISI 0-2: No inversion / well-mixed
    - AISI 2-5: Mild inversion
    - AISI 5-8: Moderate inversion, poor ventilation
    - AISI 8-10: Severe inversion, extreme trapping

    Args:
        temp_gradient: Surface inversion strength ΔT/100m (K/100m),
            positive = inversion. Values are clamped to [GRAD_MIN, GRAD_MAX].
        pbl_height_m: Planetary boundary layer height (m)
        ri_bulk: Bulk Richardson number (dimensionless, clamped to
            [RI_MIN, RI_MAX] so calm-wind noise can't dominate)
        alpha: Weight for temperature gradient term
        beta: Weight for inverse PBL height term
        gamma: Weight for Richardson number term

    Returns:
        AISI value clamped to [0, 10]
    """
    import math
    try:
        tg = float(temp_gradient)
        pblh = float(pbl_height_m)
        ri = float(ri_bulk)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(tg) and math.isfinite(pblh) and math.isfinite(ri)):
        return 0.0
    tg = max(GRAD_MIN, min(GRAD_MAX, tg))
    ri = max(RI_MIN, min(RI_MAX, ri))
    pblh_capped = max(pblh, 50.0)

    aisi = (
        alpha * tg
        + beta * (1.0 / pblh_capped)
        + gamma * ri
    )

    return max(0.0, min(AISI_MAX, aisi))


def calculate_aisi_detailed(
    temp_gradient: float,
    pbl_height_m: float,
    ri_bulk: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> dict:
    """Same as calculate_aisi but returns per-term contributions for debugging.

    Returns dict with term_grad, term_pbl, term_ri, aisi, and clamped inputs.
    Use this in tests / /aisi/current sub_terms to see which term dominates.
    """
    tg = max(GRAD_MIN, min(GRAD_MAX, float(temp_gradient or 0.0)))
    ri = max(RI_MIN, min(RI_MAX, float(ri_bulk or 0.0)))
    pblh = max(float(pbl_height_m or 700.0), 50.0)
    term_grad = alpha * tg
    term_pbl = beta * (1.0 / pblh)
    term_ri = gamma * ri
    total = max(0.0, min(AISI_MAX, term_grad + term_pbl + term_ri))
    return {
        "term_grad": round(term_grad, 3),
        "term_pbl": round(term_pbl, 3),
        "term_ri": round(term_ri, 3),
        "aisi": round(total, 2),
        "inputs_clamped": {
            "temp_gradient_k_per_100m": tg,
            "pbl_height_m": pblh,
            "ri_bulk": ri,
        },
    }


def aisi_severity_category(aisi: float) -> Tuple[str, str, str]:
    """
    Classify AISI into severity categories with descriptions and colors.

    Args:
        aisi: AISI value (0-10)

    Returns:
        Tuple of (category_name, description, hex_color)
    """
    if aisi < 2.0:
        return (
            "Well-Mixed",
            "Good ventilation, no inversion. Pollutants dispersing normally.",
            "#00e400",
        )
    elif aisi < 5.0:
        return (
            "Mild Inversion",
            "Weak inversion present. Some pollutant accumulation possible.",
            "#ffff00",
        )
    elif aisi < 8.0:
        return (
            "Moderate Inversion",
            "Significant inversion. Poor ventilation, pollutant trapping likely. "
            "Sensitive groups should limit outdoor exposure.",
            "#ff7e00",
        )
    else:
        return (
            "Severe Inversion",
            "Extreme atmospheric stagnation. Hazardous pollutant levels expected. "
            "Avoid outdoor activity. GRAP Stage III/IV activation recommended.",
            "#ff0000",
        )


def grap_activation_level(aisi: float, pm25: float = 0.0) -> dict:
    """
    Determine GRAP (Graded Response Action Plan) activation level.

    Combines AISI with PM2.5 concentration for Delhi's emergency response framework.

    Args:
        aisi: AISI value (0-10)
        pm25: Current PM2.5 concentration (µg/m³)

    Returns:
        Dict with stage, label, color, and recommended actions
    """
    # GRAP Stage determination (AQI > thresholds OR AISI > thresholds)
    if aisi >= 8.0 or pm25 >= 300:
        return {
            "stage": "IV",
            "label": "Emergency",
            "color": "#7e0023",
            "actions": [
                "Stop entry of trucks into Delhi (except essential)",
                "Ban construction & demolition activities",
                "50% staff work from home in govt offices",
                "Closure of schools (up to Class V)",
                "Water sprinkling via helicopters",
            ],
        }
    elif aisi >= 6.0 or pm25 >= 200:
        return {
            "stage": "III",
            "label": "Severe",
            "color": "#99004c",
            "actions": [
                "Ban BS-III petrol & BS-IV diesel vehicles",
                "Enhance public transport frequency",
                "Intensify road vacuum cleaning",
                "Ban stone crushers & hot-mix plants",
            ],
        }
    elif aisi >= 4.0 or pm25 >= 120:
        return {
            "stage": "II",
            "label": "Very Poor",
            "color": "#ff0000",
            "actions": [
                "Enhance CNG/electric bus deployment",
                "Strict enforcement of dust control at construction sites",
                "Increase mechanized sweeping of roads",
            ],
        }
    elif aisi >= 2.0 or pm25 >= 60:
        return {
            "stage": "I",
            "label": "Poor",
            "color": "#ff7e00",
            "actions": [
                "Enforce ban on open burning of waste",
                "Periodic mechanized sweeping on identified roads",
                "Strict enforcement of Pollution Under Control norms",
            ],
        }
    else:
        return {
            "stage": "None",
            "label": "Normal",
            "color": "#00e400",
            "actions": ["Routine monitoring and enforcement"],
        }
