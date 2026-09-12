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
from backend.data.cams_client import CAMSClient
from backend.data.openaq_client import OpenAQClient, StationReading


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

    # 4. pagination + fresh-first ordering -------------------------------
    def _loc(i, last, params=("pm25", "pm10")):
        return {
            "id": i, "name": f"station-{i}",
            "coordinates": {"latitude": 28.6, "longitude": 77.2},
            "sensors": [{"id": 100000 + i * 10 + n,
                         "parameter": {"name": p}}
                        for n, p in enumerate(params)],
            "datetimeLast": {"utc": last} if last else None,
        }

    page1 = ([_loc(10, "2018-02-22T04:00:00Z"),
              _loc(20, "2026-09-11T10:30:00Z")]
             + [_loc(1000 + i, "2019-05-01T00:00:00Z", params=())
                for i in range(98)])  # full page -> paginate on
    page2 = [_loc(999001, "2026-09-12T19:00:00Z"),
             _loc(999002, "2026-09-12T19:00:00Z")]
    pages = {"calls": []}

    async def fake_paged(endpoint, params=None, max_retries=3):
        pages["calls"].append((endpoint, (params or {}).get("page")))
        if endpoint == "locations":
            pg = (params or {}).get("page", 1)
            if pg == 1:
                return {"results": page1}
            if pg == 2:
                return {"results": page2}
            return {"results": []}
        return {"results": []}

    client2 = OpenAQClient()
    client2._rate_limited_request = fake_paged
    locs = await client2.get_locations_in_delhi(force_fresh=True)
    head = [l["id"] for l in locs][:3]
    ok &= check("both pages fetched", {10, 20, 999001, 999002}
                <= {l["id"] for l in locs},
                f"got {len(locs)} locations")
    ok &= check("fresh-first ordering",
                set(head[:2]) == {999001, 999002}, f"head={head}")
    ok &= check("dead archive sinks last", locs[-1]["id"] == 10,
                f"tail={[l['id'] for l in locs][-2:]}")

    # ... and the [:N] slice in get_latest_measurements keeps live ones.
    async def fake_latest(endpoint, params=None, max_retries=3):
        if endpoint == "locations":
            return {"results": page1 + page2}
        loc_id = endpoint.split("/")[1]
        return {"results": [{
            "measurements": [],
            "value": 42.0,
            "parameter": {"name": "pm25"},
            "datetime": {"utc": "2026-09-12T19:00:00Z"},
            "coordinates": {"latitude": 28.6, "longitude": 77.2},
            "id": loc_id, "name": f"station-{loc_id}",
        }]}

    client3 = OpenAQClient()
    client3._rate_limited_request = fake_latest
    got = await client3.get_latest_measurements(force_fresh=True)
    got_ids = {r.station_id for r in got}
    ok &= check("live locations reach /latest",
                "999001" in got_ids and "999002" in got_ids,
                f"got {sorted(got_ids)}")

    # 5. fresh-first matching --------------------------------------------
    svc.stations = [{
        "id": "t-rkp", "name": "R K Puram, Delhi - DPCC",
        "short_name": "R K Puram", "latitude": 28.5632, "longitude": 77.1869,
    }]
    old_ts = (now - timedelta(hours=33)).isoformat()
    new_ts = (now - timedelta(minutes=30)).isoformat()
    stale_official = StationReading(
        station_id="17", station_name="R K Puram, Delhi - DPCC",
        latitude=28.5632, longitude=77.1869, timestamp=old_ts,
        pollutants={"pm25": 120.0, "pm10": 200.0}, source="openaq",
        provider="CPCB")
    fresh_private = StationReading(
        station_id="4712609", station_name="Air Check",
        latitude=28.5900, longitude=77.2100, timestamp=new_ts,  # ~3.7 km
        pollutants={"pm25": 45.0, "pm10": 80.0}, source="openaq",
        provider="AirGradient")
    m_fresh = svc._match_readings([stale_official, fresh_private],
                                  fresh_within_h=6)
    ok &= check("private sensors never contribute (even fresh+close)",
                m_fresh.get("t-rkp", {}).get("pollutants", {}).get("pm25") == 120.0,
                f"got {m_fresh.get('t-rkp', {}).get('pollutants')}")
    fresh_reference = StationReading(
        station_id="8118", station_name="New Delhi",
        latitude=28.5900, longitude=77.2100, timestamp=new_ts,  # ~3.7 km
        pollutants={"pm25": 45.0, "pm10": 80.0}, source="openaq",
        provider="AirNow")
    m_freshref = svc._match_readings([stale_official, fresh_reference],
                                     fresh_within_h=6)
    ok &= check("fresh reference beats anchored-stale",
                m_freshref.get("t-rkp", {}).get("pollutants", {}).get("pm25") == 45.0,
                f"got {m_freshref.get('t-rkp', {}).get('pollutants')}")
    m_legacy = svc._match_readings([stale_official, fresh_private])
    ok &= check("legacy path keeps anchor behavior",
                m_legacy.get("t-rkp", {}).get("pollutants", {}).get("pm25") == 120.0,
                f"got {m_legacy.get('t-rkp', {}).get('pollutants')}")
    # No fresh monitor nearby -> stale reference fallback still matches.
    m_only_stale = svc._match_readings([stale_official], fresh_within_h=6)
    ok &= check("stale fallback when nothing fresh",
                m_only_stale.get("t-rkp", {}).get("pollutants", {}).get("pm25") == 120.0,
                f"got {m_only_stale.get('t-rkp')}")

    # 5b. tier-2 containment + sanity gate -------------------------------
    far_private = StationReading(
        station_id="4712610", station_name="Far Suburb Sensor",
        latitude=28.7000, longitude=77.3000, timestamp=new_ts,  # ~19 km
        pollutants={"pm25": 44.0}, source="openaq", provider="AirGradient")
    m_far = svc._match_readings([stale_official, far_private],
                                fresh_within_h=6)
    ok &= check("far private cannot smear (5 km cap)",
                m_far.get("t-rkp", {}).get("pollutants", {}).get("pm25") == 120.0,
                f"got {m_far.get('t-rkp', {}).get('pollutants')}")
    indoor = StationReading(
        station_id="4663956", station_name="Some Apartments",
        latitude=28.5650, longitude=77.1880, timestamp=new_ts,  # adjacent
        pollutants={"pm25": 3.4}, source="openaq", provider="AirGradient")
    m_indoor = svc._match_readings([stale_official, indoor],
                                   fresh_within_h=6)
    ok &= check("implausible indoor value rejected (pm25 3.4)",
                m_indoor.get("t-rkp", {}).get("pollutants", {}).get("pm25") == 120.0,
                f"got {m_indoor.get('t-rkp', {}).get('pollutants')}")
    stale_private = StationReading(
        station_id="999", station_name="Old Balcony Sensor",
        latitude=28.5650, longitude=77.1880, timestamp=old_ts,
        pollutants={"pm25": 60.0}, source="openaq", provider="AirGradient")
    m_stalepriv = svc._match_readings([stale_private, stale_official],
                                      fresh_within_h=6)
    ok &= check("stale private discarded, stale ref kept",
                m_stalepriv.get("t-rkp", {}).get("pollutants", {}).get("pm25") == 120.0,
                f"got {m_stalepriv.get('t-rkp', {}).get('pollutants')}")
    ok &= check("reference detection by provider",
                AQIService._is_reference_source("AirGradient") is False
                and AQIService._is_reference_source("CPCB") is True
                and AQIService._is_reference_source("AirNow") is True)
    ok &= check("reference detection by name suffix",
                AQIService._is_reference_source("unknown",
                    "R K Puram, Delhi - DPCC") is True)
    ok &= check("sanity gate",
                AQIService._sane_reading(indoor) is False
                and AQIService._sane_reading(fresh_private) is True
                and AQIService._sane_reading(stale_official) is True)

    # 5c. distance-weighted blending --------------------------------------
    svc.stations = [
        {"id": "s-north", "name": "Test Central North", "short_name": "Central N",
         "latitude": 28.6000, "longitude": 77.2000},
        {"id": "s-south", "name": "Test Central South", "short_name": "Central S",
         "latitude": 28.5650, "longitude": 77.2000},
    ]
    mon_a = StationReading(
        station_id="a", station_name="North Probe",
        latitude=28.6100, longitude=77.2000, timestamp=new_ts,  # ~1.1 km N
        pollutants={"pm25": 100.0}, source="openaq", provider="CPCB")
    mon_b = StationReading(
        station_id="b", station_name="South Probe",
        latitude=28.5600, longitude=77.2000, timestamp=new_ts,  # ~4.4 km S
        pollutants={"pm25": 40.0}, source="openaq", provider="AirNow")
    m_blend = svc._match_readings([mon_a, mon_b], fresh_within_h=6)
    v_n = m_blend.get("s-north", {}).get("pollutants", {}).get("pm25")
    v_s = m_blend.get("s-south", {}).get("pollutants", {}).get("pm25")
    ok &= check("blend leans to nearer monitor (not winner-take-all)",
                v_n is not None and 90.0 < v_n < 100.0, f"got {v_n}")
    ok &= check("nearby stations get different values",
                v_n is not None and v_s is not None and v_n != v_s
                and v_n > v_s,
                f"got north={v_n} south={v_s}")
    ok &= check("match tier recorded",
                m_blend.get("s-north", {}).get("match_tier") == "fresh-ref",
                f"got {m_blend.get('s-north', {}).get('match_tier')}")
    # Co-located monitor: exact value, no singularity.
    mon_home = StationReading(
        station_id="h", station_name="Home Probe",
        latitude=28.6000, longitude=77.2000, timestamp=new_ts,
        pollutants={"pm25": 77.0}, source="openaq", provider="IMD")
    m_home = svc._match_readings([mon_home], fresh_within_h=6)
    ok &= check("co-located monitor exact, finite",
                m_home.get("s-north", {}).get("pollutants", {}).get("pm25") == 77.0,
                f"got {m_home.get('s-north', {}).get('pollutants')}")
    # One voice per location: a 2-sensor site must not outweigh a
    # single-sensor neighbour at ~4x the distance.
    svc.stations = [{
        "id": "s-one", "name": "Lone Station", "short_name": "Lone",
        "latitude": 28.6000, "longitude": 77.2000}]
    row1 = StationReading(
        station_id="x", station_name="Busy Site A",
        latitude=28.6050, longitude=77.2000, timestamp=new_ts,
        pollutants={"pm25": 100.0}, source="openaq", provider="CPCB")
    row2 = StationReading(
        station_id="x", station_name="Busy Site A",
        latitude=28.6050, longitude=77.2000, timestamp=new_ts,
        pollutants={"pm25": 20.0}, source="openaq", provider="CPCB")
    neigh = StationReading(
        station_id="y", station_name="Far Probe",
        latitude=28.6200, longitude=77.2000, timestamp=new_ts,
        pollutants={"pm25": 40.0}, source="openaq", provider="CPCB")
    m_dup = svc._match_pool([row1, row2, neigh])
    v_dup = m_dup.get("s-one", {}).get("pollutants", {}).get("pm25")
    ok &= check("multi-sensor site counts once",
                v_dup is not None and 90.0 < v_dup < 100.0, f"got {v_dup}")

    # 6. CAMS parse: latest slot <= now, co µg/m³ -> mg/m³ ----------------
    cams = CAMSClient()
    real_now = datetime.now(timezone.utc).replace(minute=0, second=0,
                                                 microsecond=0)
    def _slot(dt):
        return dt.strftime("%Y-%m-%dT%H:%M")
    canned = {"hourly": {
        "time": [_slot(real_now - timedelta(hours=2)),
                 _slot(real_now - timedelta(hours=1)),
                 _slot(real_now + timedelta(hours=1))],
        "pm2_5": [50.0, 44.0, 40.0],
        "pm10": [90.0, 80.0, 75.0],
        "nitrogen_dioxide": [20.0, 18.0, 17.0],
        "sulphur_dioxide": [8.0, 7.0, 7.0],
        "ozone": [60.0, 55.0, 50.0],
        "carbon_monoxide": [400.0, 500.0, 600.0],
    }}

    class _FakeResp:
        status_code = 200

        def json(self):
            return canned

    class _FakeHttp:
        async def get(self, url, params=None):
            return _FakeResp()

    cams._http_client = _FakeHttp()
    datum = await cams.get_current_aq(28.56, 77.18)
    ok &= check("cams picks slot <= now",
                datum is not None and datum["pollutants"].get("pm25") == 44.0,
                f"got {datum}")
    ok &= check("cams co converted to mg/m³",
                datum is not None and datum["pollutants"].get("co") == 0.5,
                f"got {(datum or {}).get('pollutants', {}).get('co')}")
    ok &= check("cams labeled + timestamped",
                datum is not None and datum.get("source") == "cams"
                and datum.get("timestamp", "").endswith("+00:00"),
                f"got {datum}")

    # 7. CAMS override wiring: stale observed -> live model display,
    #    model values never persisted -----------------------------------
    svc2 = AQIService()
    svc2.stations = [{
        "id": "t1", "name": "Test Station", "short_name": "Test",
        "latitude": 28.56, "longitude": 77.18, "city": "Delhi",
        "zone": "Test", "type": "Test", "elevation_m": 0,
    }]
    svc2.settings.waqi_api_key = None  # isolate CAMS path
    svc2.settings.stale_after_hours = 6.0

    async def fake_latest_measurements(force_fresh=False):
        return [StationReading(
            station_id="17", station_name="Far Official",
            latitude=28.56, longitude=77.18,
            timestamp=(datetime.now(timezone.utc)
                       - timedelta(hours=33)).isoformat(),
            pollutants={"pm25": 120.0, "pm10": 200.0}, source="openaq")]

    async def fake_cams_batch(coords):
        return {"t1": {
            "pollutants": {"pm25": 40.0, "pm10": 70.0},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "cams"}}

    stored = []

    async def fake_store(records):
        stored.extend(records)

    svc2.openaq.get_latest_measurements = fake_latest_measurements
    svc2.cams.get_current_batch = fake_cams_batch
    svc2.preprocessor.store_readings = fake_store
    svc2.weather.get_current_weather = lambda *a: asyncio.sleep(0, result={})
    svc2.fire.get_active_fires = lambda *a, **k: asyncio.sleep(0, result=[])
    await svc2._collect_raw_data()
    cur = {r["station_id"]: r for r in svc2.state.get("raw_records", [])}["t1"]
    ok &= check("stale observed replaced by cams in snapshot",
                cur.get("source") == "cams"
                and cur.get("pollutants", {}).get("pm25") == 40.0,
                f"got {cur.get('source')} {cur.get('pollutants')}")
    ok &= check("cams values not persisted",
                all(r.get("source") != "cams" for r in stored),
                f"stored {len(stored)} records")
    ok &= check("cams_filled counted",
                svc2.state.get("cams_filled") == 1,
                f"got {svc2.state.get('cams_filled')}")

    await svc.close()
    await svc2.close()
    await client.close()
    await client2.close()
    await client3.close()
    await cams.close()

    print("\nALL PASS" if ok else "\nSOME FAILURES")
    return ok


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
