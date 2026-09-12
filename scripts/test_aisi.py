"""
AISI smoke test — run:  python scripts/test_aisi.py

Checks the optimised AISI stack end-to-end:
1. Pure formula: day (well-mixed ~0-2), night-mild, severe-night (8+ reachable)
2. Unit check: temperature_gradient() returns K/100m by default
3. Clamping: garbage Ri / gradient can't explode AISI
4. Service: PBLModel + AISICalculator on calm-night vs windy-day weather
5. GRAP: severe PM2.5 escalates stage even if AISI is moderate
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.formulas.aisi_formulas import (
    calculate_aisi, calculate_aisi_detailed,
    temperature_gradient, aisi_severity_category,
    grap_activation_level,
)
from backend.physics.pbl_model import PBLModel
from backend.physics.aisi_calculator import AISICalculator


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    return cond


async def main():
    ok = True

    # 1. temperature_gradient units
    g = temperature_gradient(20.0, 21.5, 100.0)  # +1.5K over 100m
    ok &= check("gradient K/100m", abs(g - 1.5) < 1e-9, f"got {g}")
    g_m = temperature_gradient(20.0, 21.5, 100.0, per_100m=False)
    ok &= check("gradient K/m opt-out", abs(g_m - 0.015) < 1e-9, f"got {g_m}")

    # 2. Formula bands
    day = calculate_aisi(-0.98, 1200, -0.1)
    ok &= check("day well-mixed 0-2", 0 <= day <= 2.5, f"got {day:.2f}")
    night_mild = calculate_aisi(1.5, 350, 0.3)
    ok &= check("calm night mild/moderate 2-8",
                2 <= night_mild <= 8, f"got {night_mild:.2f}")
    severe = calculate_aisi(3.5, 120, 0.8)
    ok &= check("severe night 8-10", severe >= 8, f"got {severe:.2f}")
    print("   detail severe:", calculate_aisi_detailed(3.5, 120, 0.8))

    # 3. Clamping — noisy inputs can't explode / NaN
    wild = calculate_aisi(50.0, 10.0, 99.0)
    ok &= check("wild inputs clamped <=10", 0 <= wild <= 10, f"got {wild}")
    neg_ri = calculate_aisi(-0.98, 1200, -5.0)
    ok &= check("unstable Ri clamped >=0", neg_ri >= 0, f"got {neg_ri:.2f}")
    nan = calculate_aisi(float("nan"), 700, 0.1)
    ok &= check("NaN -> 0.0", nan == 0.0, f"got {nan}")

    # 4. Service: calm night vs windy afternoon
    pbl = PBLModel()
    calm_night = {"temperature_c": 8.0, "pressure_hpa": 1015,
                  "relative_humidity": 85, "wind_speed_ms": 0.8,
                  "wind_direction_deg": 315, "pbl_height_m": 150}
    windy_day = {"temperature_c": 24.0, "pressure_hpa": 1010,
                 "relative_humidity": 30, "wind_speed_ms": 5.0,
                 "wind_direction_deg": 270, "pbl_height_m": 1400}
    pn = await pbl.compute(calm_night, hour_ist=2)
    pd = await pbl.compute(windy_day, hour_ist=14)
    print(f"   calm-night PBL: {pn}")
    print(f"   windy-day PBL:  {pd}")
    ok &= check("night inversion > day", pn["inversion_strength_k"] > pd["inversion_strength_k"])
    ok &= check("night PBL < day PBL", pn["pbl_height_m"] < pd["pbl_height_m"])

    calc = AISICalculator()
    an = await calc.compute(calm_night, hour_ist=2, pm25=280)
    calc2 = AISICalculator()  # fresh (no EMA carryover)
    ad = await calc2.compute(windy_day, hour_ist=14, pm25=60)
    print(f"   calm-night AISI: {an['aisi']} {an['category']} terms={an['sub_terms']}")
    print(f"   windy-day AISI:  {ad['aisi']} {ad['category']} terms={ad['sub_terms']}")
    ok &= check("night AISI > day AISI", an["aisi"] > ad["aisi"])
    ok &= check("day AISI well-mixed/mild", ad["aisi"] < 5, f"got {ad['aisi']}")

    # 5. GRAP reconciles with PM2.5 (no Moderate+Stage-II mismatch)
    g1 = grap_activation_level(aisi=4.0, pm25=320)
    ok &= check("high PM escalates GRAP", g1["stage"] in ("III", "IV"), f"got {g1['stage']}")
    cat, _, _ = aisi_severity_category(an["aisi"])
    ok &= check("category non-empty", bool(cat), cat)

    print("\n" + ("ALL PASS" if ok else "SOME FAILS -- see above"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
