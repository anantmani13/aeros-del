import asyncio
import sys

sys.path.insert(0, ".")
import httpx

VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_direction_10m", "surface_pressure", "cloud_cover",
        "boundary_layer_height"]


async def main():
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": 28.6139, "longitude": 77.2090,
                    "start_date": "2026-08-13", "end_date": "2026-09-12",
                    "hourly": ",".join(VARS), "timezone": "UTC"})
        print("status:", r.status_code)
        d = r.json()
        h = d.get("hourly", {})
        print("keys:", sorted(h.keys()))
        print("n_hours:", len(h.get("time", [])))
        print("first:", h.get("time", ["?"])[0],
              "| last:", h.get("time", ["?"])[-1])
        for k in VARS:
            vals = [v for v in h.get(k, []) if v is not None]
            print(f"{k}: {len(vals)}/{len(h.get('time', []))} present")


asyncio.run(main())
