"""
Backfill fire history from NASA FIRMS into SQLite fire_data.

  python scripts/backfill_fires.py [--days 7] [--dry-run]

How it works (and its honest limits):
  - FIRMS NRT API only goes back 5 days and takes no date parameter
    (verified: "Invalid day range. Expects [1..5]").
  - FIRMS monthly archives need an Earthdata login + email workflow —
    not scriptable, so not used.
  - What IS scriptable: the rolling 7-day global VIIRS CSV
    (SUOMI_VIIRS_C2_Global_7d.csv, ~35MB), filtered to the
    Punjab/Haryana/Delhi/W-UP bbox. So fire history covers the most
    recent 7 days — which is exactly where the training holdout lives.
  - Pre-season (Aug): fires were sparse anyway; the per-day counts in
    the report show this. October burning season accumulates live.

Stores: latitude, longitude, frp, confidence, acq_date (UTC ISO),
region, distance_to_delhi_km. Report: scripts/fire_backfill_report.json.
"""
import argparse
import asyncio
import csv
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.fire_client import FireClient

VIIRS_7D_URL = ("https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
                "suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_7d.csv")
BBOX = {"lon_min": 74.5, "lon_max": 78.0,
        "lat_min": 27.0, "lat_max": 32.5}


def parse_acq_utc(date_s: str, time_s: str):
    try:
        t = (time_s or "0000").strip().zfill(4)
        return datetime.strptime(f"{date_s.strip()} {t}", "%Y-%m-%d %H%M"
                                 ).replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", default="data/aqi_data.db")
    args = ap.parse_args()

    import httpx
    tmp = Path(tempfile.gettempdir()) / "SUOMI_VIIRS_7d.csv"
    print(f"Downloading VIIRS 7-day global (~35MB)...", flush=True)
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as c:
        async with c.stream("GET", VIIRS_7D_URL) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                async for chunk in r.aiter_bytes(1 << 20):
                    f.write(chunk)
    print(f"Saved {tmp} ({tmp.stat().st_size // 1_000_000}MB)", flush=True)

    fc = FireClient()
    rows, per_day = [], Counter()
    with open(tmp, newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                frp = float(row.get("frp") or 0)
            except (ValueError, TypeError, KeyError):
                continue
            if not (BBOX["lon_min"] <= lon <= BBOX["lon_max"]
                    and BBOX["lat_min"] <= lat <= BBOX["lat_max"]):
                continue
            acq = parse_acq_utc(row.get("acq_date", ""), row.get("acq_time", ""))
            if acq is None:
                continue
            age_days = (datetime.now(timezone.utc) - acq).days
            if age_days > args.days:
                continue
            conf = str(row.get("confidence", "nominal")).lower()
            rows.append({
                "latitude": lat, "longitude": lon, "frp": frp,
                "confidence": conf,
                "acq_date": acq.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "region": fc._classify_region(lat, lon),
                "distance_to_delhi_km": fc._haversine_distance(
                    lat, lon, 28.6139, 77.2090),
            })
            per_day[acq.strftime("%Y-%m-%d")] += 1
    tmp.unlink(missing_ok=True)

    total_frp = round(sum(r["frp"] for r in rows), 1)
    print(f"Matches in bbox, last {args.days}d: {len(rows)} "
          f"(FRP {total_frp} MW)")
    for day in sorted(per_day):
        print(f"  {day}: {per_day[day]} fires")

    inserted = 0
    if rows and not args.dry_run:
        con = sqlite3.connect(str(PROJECT_ROOT / args.db))
        con.execute("DELETE FROM fire_data WHERE substr(acq_date,1,10) >= "
                    "date('now','-8 days')")
        con.executemany(
            "INSERT INTO fire_data (latitude, longitude, frp, confidence,"
            " acq_date, region, distance_to_delhi_km)"
            " VALUES (:latitude, :longitude, :frp, :confidence,"
            " :acq_date, :region, :distance_to_delhi_km)", rows)
        inserted = con.total_changes
        con.commit()
        con.close()
        print(f"Inserted: {inserted} (refreshed last-8d window)")
    report = {"days": args.days, "bbox": BBOX, "dry_run": args.dry_run,
              "fires": len(rows), "total_frp_mw": total_frp,
              "inserted": inserted, "per_day": dict(sorted(per_day.items()))}
    out = PROJECT_ROOT / "scripts" / "fire_backfill_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Report: {out}")


if __name__ == "__main__":
    asyncio.run(main())
