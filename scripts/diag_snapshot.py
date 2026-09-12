"""Inspect a fetched snapshot JSON for suspicious AQI values."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "snap.json"
d = json.load(open(path, encoding="utf-8"))
print("last_update:", d.get("last_update"), "| source:", d.get("data_source"))
print("domain:", json.dumps(d.get("domain_summary"), indent=1)[:800])
print()
for s in d.get("stations", []):
    c = s.get("current") or {}
    p = c.get("pollutants") or {}
    print(
        f"{s.get('short_name')}: aqi={c.get('aqi')} cat={c.get('category')} "
        f"src={c.get('source')} age_h={c.get('age_hours')} "
        f"ts={c.get('timestamp')} dom={c.get('dominant_pollutant')} "
        f"subs={c.get('sub_indices')} pol={p}"
    )
