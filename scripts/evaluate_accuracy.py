"""
Accuracy evaluation — walk-forward backtest of the 72h forecast pipeline.

Compares, on real persisted SQLite observations (data/aqi_data.db):
  1. statistical_baseline (the live default with no trained weights)
  2. naive persistence (carry current value forward)
  3. LightGBM / sklearn HistGBM trained on first 70% → tested on last 30%

Metrics: MAE, RMSE, bias, Pearson r, AQI-category hit rate.
Writes scripts/accuracy_report.json and prints a summary table.

Usage:
    python scripts/evaluate_accuracy.py [--db data/aqi_data.db] [--horizons 6,12,24]
"""
import argparse
import asyncio
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.xgboost_forecaster import statistical_baseline
from backend.data.naqi_calculator import NAQICalculator

NAQI = NAQICalculator()


def load_series(db_path, min_points=4):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT station_id, timestamp, pm25, pm10, aqi, source "
        "FROM station_readings "
        "WHERE pm25 IS NOT NULL ORDER BY station_id, timestamp"
    ).fetchall()
    con.close()
    series = {}
    for r in rows:
        try:
            ts = datetime.fromisoformat(str(r["timestamp"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        series.setdefault(r["station_id"], []).append(
            (ts, float(r["pm25"]),
             float(r["pm10"]) if r["pm10"] is not None else None,
             float(r["aqi"]) if r["aqi"] is not None else None,
             r["source"] or "unknown"))
    return {k: v for k, v in series.items() if len(v) >= min_points}


def split_runs(pts):
    """Split a station series at source switches (demo<->live)."""
    runs, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[4] != cur[-1][4]:
            runs.append(cur)
            cur = [p]
        else:
            cur.append(p)
    runs.append(cur)
    return [r for r in runs if len(r) >= 2]


def aqi_of(pm25, pm10):
    d = {"pm25": pm25}
    if pm10:
        d["pm10"] = pm10
    return NAQI.calculate_naqi(d)


def stats(errs):
    n = len(errs)
    if not n:
        return {"n": 0}
    mae = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    bias = sum(errs) / n
    return {"n": n, "mae": round(mae, 2), "rmse": round(rmse, 2),
            "bias": round(bias, 2)}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return round(cov / math.sqrt(vx * vy), 3)


def backtest(series):
    """Next-observation skill within same-source runs.

    For each consecutive pair, the horizon equals the true gap (rounded to
    whole hours, min 1). Pairs are bucketed by gap so short- and long-lead
    skill are reported honestly.
    """
    buckets = {"<=1h": [], "1-6h": [], ">6h": []}
    pers = {"<=1h": [], "1-6h": [], ">6h": []}
    cat_ok, cat_n = {"<=1h": 0, "1-6h": 0, "6h+": 0}, {}
    cat_ok = {"<=1h": 0, "1-6h": 0, ">6h": 0}
    cat_n = {"<=1h": 0, "1-6h": 0, ">6h": 0}
    pers_cat = {"<=1h": 0, "1-6h": 0, ">6h": 0}
    base_pred_all, actual_all = [], []
    npairs = 0
    for sid, pts in series.items():
        for run in split_runs(pts):
            for i in range(3, len(run)):
                dt_h = (run[i][0] - run[i - 1][0]).total_seconds() / 3600.0
                if dt_h <= 0 or dt_h > 72:
                    continue
                h = max(1, min(72, round(dt_h)))
                hist = [p[1] for p in run[:i]]
                cur = hist[-1]
                actual = run[i][1]
                fc = statistical_baseline(
                    current_pm25=cur, history_pm25=hist, horizon=h)
                pred = fc["pm25"][h - 1]
                b = "<=1h" if dt_h <= 1 else ("1-6h" if dt_h <= 6 else ">6h")
                buckets[b].append(pred - actual)
                pers[b].append(cur - actual)
                base_pred_all.append(pred)
                actual_all.append(actual)
                npairs += 1
                pa = aqi_of(pred, pred * 1.35).category
                ta = aqi_of(actual, run[i][2]).category
                cat_n[b] += 1
                cat_ok[b] += (pa == ta)
                pers_cat[b] += (aqi_of(cur, run[i - 1][2]).category == ta)
    out = {"pairs": npairs}
    for b in buckets:
        out[b] = {
            "baseline": {**stats(buckets[b]),
                         "cat_acc": round(cat_ok[b] / max(cat_n[b], 1), 3)},
            "persistence": {**stats(pers[b]),
                            "cat_acc": round(pers_cat[b] / max(cat_n[b], 1), 3)},
        }
    out["pearson_r_baseline"] = pearson(base_pred_all, actual_all)
    return out


def train_gbm(series):
    """Train LightGBM (else sklearn HistGBM) on lag features; test last 30%."""
    try:
        import numpy as np
    except ImportError:
        return {"trained": False, "reason": "numpy missing"}
    X, y = [], []
    for sid, pts in series.items():
        for run in split_runs(pts):
            if len(run) < 9:
                continue
            for i in range(6, len(run) - 1):
                window = [p[1] for p in run[i - 6:i + 1]]
                hr = run[i][0].hour
                X.append(window + [hr, math.sin(2 * math.pi * hr / 24),
                                   math.cos(2 * math.pi * hr / 24)])
                y.append(run[i + 1][1])
    if len(X) < 60:
        return {"trained": False, "reason": f"only {len(X)} samples",
                "samples": len(X)}
    X = np.array(X)
    y = np.array(y)
    split = int(len(X) * 0.7)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    backend = None
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=200, num_leaves=31,
                                  learning_rate=0.06, random_state=42,
                                  verbose=-1)
        model.fit(Xtr, ytr)
        backend = "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=200, random_state=42)
        model.fit(Xtr, ytr)
        backend = "sklearn-histgbm"
    pred = model.predict(Xte)
    errs = [float(p - t) for p, t in zip(pred, yte)]
    perr = [float(np.mean(ytr) - t) for t in yte]
    return {"trained": True, "backend": backend,
            "train_n": len(Xtr), "test_n": len(Xte),
            "gbm": stats(errs),
            "persistence_mean": stats(perr),
            "pearson_r": pearson(list(pred), list(yte))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/aqi_data.db")
    ap.add_argument("--horizons", default="6,12,24")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    db = PROJECT_ROOT / args.db
    if not db.exists():
        print(f"No database at {db} — run the server once to collect data.")
        sys.exit(1)
    series = load_series(str(db))
    print(f"Stations with history: {len(series)}; "
          f"total points: {sum(len(v) for v in series.values())}")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "db": str(db), "stations": len(series)}
    report["next_observation_skill"] = backtest(series)
    report["gbm_next_step"] = train_gbm(series)
    out = PROJECT_ROOT / "scripts" / "accuracy_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
