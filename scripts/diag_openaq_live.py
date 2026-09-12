"""
Live OpenAQ diagnostic — run:  python scripts/diag_openaq_live.py

Hits the REAL OpenAQ API with the repo's key and reports:
1. How many locations per page, and the freshness distribution.
2. Which locations survive the PM filter + [:60] slice (params shown).
3. What get_latest_measurements actually returns (count + age distribution).
4. How many repo stations match, and from how far / how old.
"""
import asyncio
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.service import AQIService


def age_h(ts):
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return None


async def main():
    svc = AQIService()
    client = svc.openaq

    locs = await client.get_locations_in_delhi(force_fresh=True)
    print(f"LOCATIONS total={len(locs)}")
    buckets = Counter()
    for loc in locs:
        a = age_h(loc.get("last_updated"))
        key = ("null" if a is None else "<6h" if a < 6 else "<24h" if a < 24
               else "<48h" if a < 48 else "<7d" if a < 168 else "older")
        buckets[key] += 1
    print("  last_updated age distribution:", dict(buckets))

    pm = [loc for loc in locs
          if any(p in ("pm25", "pm10") for p in (loc.get("parameters") or []))]
    print(f"PM-capable total={len(pm)}; sliced to {min(60, len(pm))}")
    print("  top-15 sliced (fresh-first):")
    for loc in pm[:15]:
        a = age_h(loc.get("last_updated"))
        print(f"    id={loc['id']} age={a if a is None else round(a,1)}h "
              f"params={loc.get('parameters')} name={loc.get('name')}")
    print("  bottom-5 sliced:")
    for loc in pm[:60][-5:]:
        a = age_h(loc.get("last_updated"))
        print(f"    id={loc['id']} age={a if a is None else round(a,1)}h "
              f"params={loc.get('parameters')} name={loc.get('name')}")

    readings = await client.get_latest_measurements(force_fresh=True)
    print(f"READINGS total={len(readings)}")
    rb = Counter()
    for r in readings:
        a = age_h(r.timestamp)
        key = ("null" if a is None else "<6h" if a < 6 else "<24h" if a < 24
               else "<48h" if a < 48 else "<7d" if a < 168 else "older")
        rb[key] += 1
    print("  measurement age distribution:", dict(rb))

    mapped = svc._match_readings(readings)
    print(f"MAPPED stations={len(mapped)}/{len(svc.stations)}")
    ages = sorted(
        (age_h(v["timestamp"]) for v in mapped.values()), key=lambda x: (x is None, x))
    show = [round(a, 1) if a is not None else None for a in ages]
    print(f"  matched reading ages (h): {show[:10]} ... {show[-5:]}")
    dists = sorted(v.get("matched_distance_km", -1) for v in mapped.values())
    print(f"  matched distances (km): {dists[:10]} ... {dists[-5:]}")

    await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
