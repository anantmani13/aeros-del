"""
Backfill hourly history from OpenAQ into SQLite — the legitimate path to
trained weights.

  python scripts/backfill_history.py --days 30
  python scripts/backfill_history.py --days 30 --dry-run   # plan only
  python scripts/backfill_history.py --days 14 --pollutants pm25,pm10

What it does:
  1. Lists OpenAQ Delhi NCR locations, keeps fresh ones (seen in last 4d)
  2. Greedily matches one OpenAQ location per stations.json station
     (haversine + name-anchor, same rule as the live matcher)
  3. Pulls 15-min raw sensor measurements for the window, resamples to
     HOURLY means, validates bounds, scores NAQI, stores to station_readings
  4. Writes scripts/backfill_report.json (per-station hours + continuity)

After it runs:  python scripts/train_models.py --force-sklearn

Needs OPENAQ_API_KEY in .env (v3 measurements endpoint requires it).
"""
import argparse
import asyncio
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import Settings, DATA_DIR
from backend.data.openaq_client import OpenAQClient, POLLUTANT_BOUNDS
from backend.data.preprocessor import DataPreprocessor
from backend.data.naqi_calculator import NAQICalculator

STOPWORDS = {"delhi", "new", "sector", "sec", "phase", "gram", "nagar",
             "marg", "road", "rd", "station", "dpcc", "cpcb", "uppcb",
             "hspcb", "imd", "iitm", "sai", "teri"}


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def name_keys(station) -> list:
    out = []
    for k in (station.get("short_name", ""), station.get("name", "")):
        k = re.sub(r"[^a-z0-9 ]", " ", str(k).lower())
        for tok in k.split():
            if len(tok) >= 4 and tok not in STOPWORDS:
                out.append(tok)
    return out


def last_utc(loc) -> str:
    dl = loc.get("datetimeLast")
    if isinstance(dl, dict):
        return dl.get("utc") or ""
    return dl or ""


def hour_key(utc_str: str) -> str:
    dt = datetime.fromisoformat(str(utc_str).replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--pollutants", default="pm25,pm10")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", default="data/aqi_data.db")
    args = ap.parse_args()
    pollutants = [p.strip().lower() for p in args.pollutants.split(",") if p.strip()]

    settings = Settings.from_env()
    stations = json.loads((DATA_DIR / "stations.json").read_text())["stations"]
    print(f"Our stations: {len(stations)}")

    client = OpenAQClient(api_key=settings.openaq_api_key, rate_limit=0.2,
                          max_concurrency=4)
    raw = await client._rate_limited_request("locations", {
        "bbox": "76.8,28.3,77.5,28.9", "limit": 100, "page": 1,
        "order_by": "id", "sort_order": "asc"})
    locations = (raw or {}).get("results", [])
    cutoff = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    fresh = [loc for loc in locations if last_utc(loc) >= cutoff]
    print(f"OpenAQ locations: {len(locations)}, fresh: {len(fresh)}")
    if not fresh:
        print("No fresh locations — check API key / network.")
        return

    # Greedy unique matching: best (station, location) pairs first
    pairs = []
    for st in stations:
        keys = name_keys(st)
        for loc in fresh:
            coords = loc.get("coordinates") or {}
            la, lo = coords.get("latitude"), coords.get("longitude")
            if la is None or lo is None:
                continue
            d = haversine(st["latitude"], st["longitude"], la, lo)
            anchored = any(k in str(loc.get("name", "")).lower() for k in keys)
            pairs.append((d * 0.5 if anchored else d, st["id"], loc))
    pairs.sort(key=lambda p: p[0])
    assigned, used_locs = {}, set()
    for eff, sid, loc in pairs:
        if sid in assigned or loc["id"] in used_locs or eff > 25.0:
            continue
        assigned[sid] = (loc, eff)
    print(f"Matched stations: {len(assigned)}")
    for sid in sorted(assigned):
        loc, eff = assigned[sid]
        print(f"  {sid:18s} <- OpenAQ {loc['id']} ({eff:.1f} km)")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    dt_from = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    dt_to = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    per_station_hours = {}
    total_inserted = 0
    if not args.dry_run:
        pre = DataPreprocessor(database_path=args.db)
        await pre.initialize_database()
        naqi = NAQICalculator()

    for sid, (loc, eff) in sorted(assigned.items()):
        sensors = {}
        for s in loc.get("sensors") or []:
            pname = ((s.get("parameter") or {}).get("name") or "").lower()
            if pname in pollutants and pname not in sensors:
                sensors[pname] = s["id"]
        if "pm25" not in sensors:
            continue
        hourly = defaultdict(dict)  # hour -> {pollutant: [values]}
        for pname, sensor_id in sensors.items():
            print(f"  {sid}: fetching {pname} (sensor {sensor_id})...",
                  flush=True)
            try:
                ms = await client.get_sensor_measurements(
                    sensor_id, dt_from, dt_to)
                print(f"  {sid}: {pname} -> {len(ms)} raw points", flush=True)
            except Exception as e:
                print(f"  {sid}: sensor {sensor_id} failed: {e}", flush=True)
                continue
            for m in ms:
                try:
                    hourly[hour_key(m["utc"])][pname].append(m["value"])
                except KeyError:
                    hourly[hour_key(m["utc"])][pname] = [m["value"]]
        records = []
        for hour in sorted(hourly):
            pol = {p: round(sum(v) / len(v), 2)
                   for p, v in hourly[hour].items() if v}
            if pol.get("pm25") is None:
                continue
            if not all(POLLUTANT_BOUNDS.get(p, (0, 1e9))[0] <= v
                       <= POLLUTANT_BOUNDS.get(p, (0, 1e9))[1]
                       for p, v in pol.items()):
                continue
            res = naqi.calculate_naqi(pol)
            records.append({
                "station_id": sid, "timestamp": hour,
                "pollutants": pol, "aqi": res.overall_aqi,
                "category": res.category,
                "dominant_pollutant": res.dominant_pollutant,
                "source": "openaq",
            })
        per_station_hours[sid] = len(records)
        if records and not args.dry_run:
            await pre.store_readings(records)
            total_inserted += len(records)
        print(f"  {sid:18s}: {len(records)} hourly rows")

    await client.close()
    total_hours = sum(per_station_hours.values())
    print(f"\n{'DRY-RUN ' if args.dry_run else ''}Total hourly rows: "
          f"{total_hours} across {len(per_station_hours)} stations")
    if not args.dry_run:
        print(f"Inserted: {total_inserted}")
        print("Next: python scripts/train_models.py --force-sklearn")
    report = {
        "days": args.days, "pollutants": pollutants,
        "dry_run": args.dry_run,
        "matched_stations": len(assigned),
        "total_hourly_rows": total_hours,
        "inserted": total_inserted,
        "per_station_hours": per_station_hours,
    }
    out = PROJECT_ROOT / "scripts" / "backfill_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Report: {out}")


if __name__ == "__main__":
    asyncio.run(main())
