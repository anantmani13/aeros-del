"""
Station freshness smoke test — run:  python scripts/test_station_freshness.py

Covers the manual-refresh / stale-badge fix:
1. _reading_age_hours: known timestamps -> expected ages (fixed clock).
2. _station_payload: fresh reading -> stale=False; 35h-old CPCB-style
   reading -> stale=True with age_hours attached.
3. OpenAQ cache bypass: with force_fresh=True a stale in-memory entry is
   ignored and the API is hit again (offline — HTTP layer monkeypatched).
"""
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.service import AQIService
from backend.data.openaq_client import OpenAQClient


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    return bool(cond)


async def main():
    ok = True
    now = datetime(2026, 9, 12, 19, 30, tzinfo=timezone.utc)

    # 1. age math -------------------------------------------------------
    age = AQIService._reading_age_hours("2026-09-12T19:00:00Z", now=now)
    ok &= check("age 30m", age is not None and abs(age - 0.5) < 1e-9, f"got {age}")
    age = AQIService._reading_age_hours("2026-09-11T08:15:00Z", now=now)
    ok &= check("age ~35h", age is not None and abs(age - 35.25) < 1e-9, f"got {age}")
    ok &= check("age garbage -> None",
                AQIService._reading_age_hours("not-a-time", now=now) is None)
    ok &= check("age empty -> None",
                AQIService._reading_age_hours(None, now=now) is None)

    # 2. payload stale flag ---------------------------------------------
    svc = AQIService()
    fresh = dict(svc._station_payload({
        "station_id": "x", "station_name": "X",
        "timestamp": (now - timedelta(minutes=20)).isoformat(),
        "pollutants": {"pm25": 80.0, "pm10": 150.0}, "source": "live",
    }))
    ok &= check("fresh payload has age", fresh.get("age_hours") is not None,
                f"got {fresh.get('age_hours')}")
    # (timestamp is "now" relative to real clock here, so it must be fresh)
    now_iso = datetime.now(timezone.utc).isoformat()
    fresh2 = svc._station_payload({
        "station_id": "x", "station_name": "X", "timestamp": now_iso,
        "pollutants": {"pm25": 80.0, "pm10": 150.0}, "source": "live",
    })
    ok &= check("fresh -> stale False", fresh2.get("stale") is False,
                f"got {fresh2.get('stale')}")
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=35)).isoformat()
    stale = svc._station_payload({
        "station_id": "x", "station_name": "X", "timestamp": stale_ts,
        "pollutants": {"pm25": 64.0, "pm10": 149.0}, "source": "live",
    })
    ok &= check("35h-old -> stale True", stale.get("stale") is True,
                f"got {stale.get('stale')}")
    ok &= check("35h-old age_hours ~35",
                stale.get("age_hours") is not None
                and abs(stale["age_hours"] - 35.0) < 0.2,
                f"got {stale.get('age_hours')}")

    # 3. force_fresh bypasses the in-memory cache ------------------------
    client = OpenAQClient()
    client._set_cached("latest_all_delhi", ["CACHED-STALE"])
    normal = await client.get_latest_measurements()
    ok &= check("normal path serves cache", normal == ["CACHED-STALE"],
                f"got {normal}")

    calls = {"n": 0}

    async def fake_request(endpoint, params=None, max_retries=3):
        calls["n"] += 1
        if endpoint == "locations":
            return {"results": []}
        return {"results": []}

    client._rate_limited_request = fake_request
    fresh_data = await client.get_latest_measurements(force_fresh=True)
    ok &= check("force_fresh hits API", calls["n"] > 0,
                f"api calls={calls['n']}")
    ok &= check("force_fresh ignores cache", fresh_data != ["CACHED-STALE"],
                f"got {fresh_data}")

    await svc.close()

    print("\nALL PASS" if ok else "\nSOME FAILURES")
    return ok


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
