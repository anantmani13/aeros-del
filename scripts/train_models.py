"""
Train ensemble weights from persisted SQLite history.

  python scripts/train_models.py
  python scripts/train_models.py --force-sklearn   # Render-compatible weights
                                                   # (no lightgbm wheel needed)

What it does:
  1. Loads per-station pollutant history from data/aqi_data.db
  2. Builds FeatureEngineer vectors per station-hour -> next-hour PM2.5
  3. Time-split 80/20, trains LightGBM (or sklearn HistGBM) + XGBoost (if
     installed), compares against persistence on the holdout
  4. SAVES NOTHING unless the model beats persistence (test MAE lower AND
     Pearson r > 0.2 AND >= 30 test rows). This guardrail is why the repo
     ships without weights today: with ~5 readings/station the gates fail,
     and shipping overfit weights would be dishonest (and verifiable via
     /api/v1/accuracy/summary).

Saved files (auto-loaded at boot by EnsembleForecaster._load_weights):
  backend/models/lgbm_pm25.joblib
  backend/models/xgboost_pm25.joblib   (only if xgboost installed)

TFT (torch) is intentionally skipped until hourly continuity exists:
sequences need consecutive hourly rows; day-apart refreshes can't train it.
"""
import argparse
import asyncio
import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.lightgbm_forecaster import LightGBMForecaster
from backend.ml.xgboost_forecaster import XGBoostForecaster
from backend.data.weather_client import WeatherClient
from backend.physics.pbl_model import PBLModel
from backend.formulas.aisi_formulas import calculate_aisi

MIN_PRIOR_POINTS = 12   # need some lag context before a row becomes a sample
MIN_SAMPLES = 300       # below this, trees just memorize noise
MIN_TEST_ROWS = 30
MIN_R = 0.2


def load_readings(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT station_id, timestamp, pm25, pm10, no2, so2, o3, co "
        "FROM station_readings WHERE pm25 IS NOT NULL "
        "ORDER BY station_id, timestamp"
    ).fetchall()
    con.close()
    series = {}
    for r in rows:
        try:
            ts = datetime.fromisoformat(str(r["timestamp"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        series.setdefault(r["station_id"], []).append({
            "timestamp": ts,
            "pollutants": {
                "pm25": r["pm25"], "pm10": r["pm10"], "no2": r["no2"],
                "so2": r["so2"], "o3": r["o3"], "co": r["co"],
            },
        })
    return series


async def ensure_weather_cache(stations, start, end, cache_path):
    """Fetch-once ERA5 archive per station; reuse until the span grows."""
    from backend.app.config import DATA_DIR as _DD
    cache_file = PROJECT_ROOT / cache_path
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            cache = {}
    have = set()
    for v in cache.values():
        have.update(v.keys())
    need_start = start.strftime("%Y-%m-%dT%H:00:00Z")
    need_end = end.strftime("%Y-%m-%dT%H:00:00Z")
    if cache and min(have, default="~") <= need_start \
            and max(have, default="") >= need_end:
        return cache, False
    client = WeatherClient()
    fresh = await client.get_history_batch(
        [{"id": s["id"], "latitude": s["latitude"],
          "longitude": s["longitude"]} for s in stations],
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    await client.close()
    cache.update(fresh)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return cache, True


def load_fires(db_path):
    """All persisted fires, sorted: [(utc, frp, dist_km)]."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT acq_date, frp, distance_to_delhi_km FROM fire_data "
        "ORDER BY acq_date").fetchall()
    con.close()
    out = []
    for acq, frp, dist in rows:
        try:
            ts = datetime.fromisoformat(str(acq).replace("Z", "+00:00"))
        except ValueError:
            continue
        out.append((ts, float(frp or 0), float(dist or 999)))
    return out


TFT_WINDOW = 72
TFT_HORIZON = 72
TFT_MIN_SEQS = 100


def build_tft_sequences(series, window=TFT_WINDOW, horizon=TFT_HORIZON):
    """Consecutive-hourly (window + horizon) sequences for TFT training.

    5 features per hour (must match _TFTEncoderDecoder.INPUT_DIM_DEFAULT,
    which load() rebuilds): [pm25, pm10, hour_sin, hour_cos, is_night].
    Raw concentration scale (no normalization) to match _torch_predict,
    which tiles raw history at inference time.
    """
    import math
    seqs = []
    for sid, pts in series.items():
        # Split into consecutive-hourly runs (gap <= 1.5h)
        runs, cur = [], [pts[0]] if pts else []
        for p in pts[1:]:
            gap = (p["timestamp"] - cur[-1]["timestamp"]).total_seconds() / 3600
            if 0 < gap <= 1.5:
                cur.append(p)
            else:
                runs.append(cur)
                cur = [p]
        if cur:
            runs.append(cur)
        for run in runs:
            if len(run) < window + horizon:
                continue
            feats = []
            ok = True
            for p in run:
                pol = p["pollutants"]
                if pol.get("pm25") is None:
                    ok = False
                    break
                hr = p["timestamp"].hour + p["timestamp"].minute / 60.0
                feats.append([
                    float(pol["pm25"]), float(pol.get("pm10") or 0.0),
                    math.sin(2 * math.pi * hr / 24),
                    math.cos(2 * math.pi * hr / 24),
                    1.0 if (hr >= 19 or hr < 7) else 0.0,
                ])
            if not ok:
                continue
            import numpy as np
            pm = [f[0] for f in feats]
            for i in range(len(feats) - window - horizon + 1):
                seqs.append({
                    "station_id": sid,
                    "end_time": run[i + window - 1]["timestamp"],
                    "history": np.array(feats[i:i + window], dtype=float),
                    "targets": [float(v) for v in
                                pm[i + window:i + window + horizon]],
                })
    seqs.sort(key=lambda s: s["end_time"])
    return seqs


async def train_tft_gated(seqs, test_cap=250):
    """Train TFT on first 80% (time), gate on 24h-ahead skill vs persistence.

    Saves backend/models/tft_pm25.pt ONLY if 24h-ahead MAE beats
    carry-forward persistence with r > 0.2 — the same honesty rule as
    the tree models. A 72h model is fairly judged away from the 1h
    horizon where persistence is near-optimal.
    """
    from backend.ml.transformer_forecaster import TFTForecaster
    split = int(len(seqs) * 0.8)
    tr, te = seqs[:split], seqs[split:]
    te = te[:test_cap]
    if len(te) < MIN_TEST_ROWS:
        return {"trained": False, "saved": False,
                "reason": f"only {len(te)} test seqs"}
    fc = TFTForecaster(model_path=None)
    rep = await fc.train(tr)
    if not rep.get("trained"):
        return {**rep, "saved": False}
    errs, perrs, preds24, acts24 = [], [], [], []
    for s in te:
        hist = [float(v) for v in s["history"][:, 0]]
        cur = hist[-1]
        try:
            out = fc._torch_predict(hist, cur, TFT_HORIZON)
        except Exception:
            continue
        actual = float(s["targets"][23])
        errs.append(out[23] - actual)
        perrs.append(cur - actual)
        preds24.append(out[23])
        acts24.append(actual)
    if not errs:
        return {"trained": True, "saved": False, "reason": "eval failed"}
    m, p = stats(errs), stats(perrs)
    r = round(pearson(preds24, acts24), 3)
    out = {"trained": True, "backend": "torch",
           "h24_mae": m["mae"], "h24_rmse": m["rmse"],
           "persist_h24_mae": p["mae"], "pearson_r_h24": r,
           "train_n": len(tr), "test_n": len(te), "saved": False}
    if m["mae"] < p["mae"] and r > MIN_R:
        import torch
        MODELS = PROJECT_ROOT / "backend" / "models"
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(fc._model.state_dict(), str(MODELS / "tft_pm25.pt"))
        out.update({"saved": True, "path": "backend/models/tft_pm25.pt"})
        out["verdict"] = (f"SAVED tft_pm25.pt — H+24 MAE {m['mae']} vs "
                          f"persist {p['mae']}, r={r}")
    else:
        out["verdict"] = (f"NOT saved — H+24 MAE {m['mae']} vs persist "
                          f"{p['mae']}, r={r}")
    return out


def fire_summary_at(fires, end_ts, window_h=48):
    """Fires in the 48h before a sample: count, FRP, nearest.

    Same shape as the live fire_stats FeatureEngineer consumes
    (total_fires / total_frp / nearest_km) — zeros when fire-blind.
    """
    from bisect import bisect_left, bisect_right
    times = [f[0] for f in fires]
    lo = bisect_left(times, end_ts - timedelta(hours=window_h))
    hi = bisect_right(times, end_ts)
    win = fires[lo:hi]
    if not win:
        return {"total_fires": 0, "total_frp": 0.0, "nearest_km": 999.0}, 0
    return {"total_fires": len(win),
            "total_frp": round(sum(f[1] for f in win), 1),
            "nearest_km": round(min(f[2] for f in win), 1)}, 1


async def build_samples(series, wx=None, fires=None):
    fe = FeatureEngineer()
    pbl_model = PBLModel()
    samples = []  # (timestamp, features, target)
    wx_hits = fire_hits = 0
    for sid, pts in series.items():
        if len(pts) < MIN_PRIOR_POINTS + 1:
            continue
        for i in range(MIN_PRIOR_POINTS, len(pts) - 1):
            window = pts[:i + 1]
            if window[-1]["pollutants"].get("pm25") is None:
                continue
            target = pts[i + 1]["pollutants"].get("pm25")
            if target is None:
                continue
            end_ts = window[-1]["timestamp"]
            hour_key = end_ts.strftime("%Y-%m-%dT%H:00:00Z")
            wx_dict, physics = None, None
            if wx and hour_key in wx.get(sid, {}):
                wx_hits += 1
                wx_dict = dict(wx[sid][hour_key])
                # Real ERA5 PBL + recomputed inversion/Ri/AISI per sample —
                # same physics path as the live server, not defaults.
                ist_hour = (end_ts.hour + 5
                            + (1 if end_ts.minute >= 30 else 0)) % 24
                pbl = await pbl_model.compute(wx_dict, hour_ist=ist_hour)
                try:
                    aisi = calculate_aisi(
                        pbl.get("inversion_strength_k", 0.0),
                        pbl.get("pbl_height_m", 700.0),
                        pbl.get("ri_bulk", 0.1))
                except Exception:
                    aisi = 2.0
                physics = {"aisi": aisi, "pbl": pbl}
            fire_dict, hit = fire_summary_at(fires or [], end_ts) \
                if fires is not None else (None, 0)
            fire_hits += hit
            feats = fe.build_features(
                station_id=sid, readings=window,
                weather=wx_dict, fire_summary=fire_dict, physics=physics,
                now=end_ts,
            )
            samples.append((pts[i + 1]["timestamp"], feats, float(target)))
    samples.sort(key=lambda s: s[0])
    n = len(samples)
    return samples, ((wx_hits / n if n else 0.0),
                     (fire_hits / n if n else 0.0))


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
    return cov / math.sqrt(vx * vy)


def stats(errs):
    n = len(errs)
    mae = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2)}


async def run_training(db: str = "data/aqi_data.db",
                       min_samples: int = MIN_SAMPLES,
                       force_sklearn: bool = False,
                       use_weather: bool = True,
                       weather_cache: str = "data/weather_history.json") -> dict:
    """Train + gate + (maybe) save ensemble weights. Returns report dict.

    Importable so the running server can auto-retrain itself
    (see AQIService.maybe_auto_train) — same code, same guardrails.
    """
    from datetime import datetime, timezone
    if force_sklearn:
        sys.modules["lightgbm"] = None  # force the sklearn fallback path

    try:
        import lightgbm  # noqa: F401
        gbm_backend = "lightgbm"
    except ImportError:
        gbm_backend = "sklearn-histgbm"
    try:
        import xgboost  # noqa: F401
        xgb_available = True
    except ImportError:
        xgb_available = False

    db = PROJECT_ROOT / db
    from datetime import datetime, timezone
    report = {"db": str(db), "gbm_backend": gbm_backend,
              "xgboost_available": xgb_available,
              "generated_at": datetime.now(timezone.utc).isoformat()}
    if not db.exists():
        report["verdict"] = f"No database at {db} — run the server first."
        print(report["verdict"])
        return report

    series = load_readings(str(db))
    n_points = sum(len(v) for v in series.values())
    print(f"Stations: {len(series)}, total readings: {n_points}")
    report.update({"stations": len(series), "readings": n_points})

    # TFT density check: needs consecutive hourly rows
    gaps_ok = gaps_all = 0
    for pts in series.values():
        for a, b in zip(pts, pts[1:]):
            gaps_all += 1
            if (b["timestamp"] - a["timestamp"]).total_seconds() / 3600 <= 2:
                gaps_ok += 1
    dense_frac = (gaps_ok / gaps_all) if gaps_all else 0
    all_ts = [p["timestamp"] for pts in series.values() for p in pts]
    span_days = ((max(all_ts) - min(all_ts)).days if len(all_ts) > 1 else 0)
    report["hourly_continuity"] = {
        "consecutive_pairs": gaps_all,
        "hourly_pairs": gaps_ok,
        "fraction": round(dense_frac, 3),
        "span_days": span_days,
    }
    try:
        import torch  # noqa: F401
        torch_available = True
    except ImportError:
        torch_available = False
    seqs = build_tft_sequences(series)
    print(f"TFT sequences (72h->72h): {len(seqs)} (need >={TFT_MIN_SEQS})"
          f" | torch: {'yes' if torch_available else 'no'}")
    report["tft_sequences"] = len(seqs)
    if len(seqs) < TFT_MIN_SEQS:
        tft_note = (f"deferred: {len(seqs)} sequences < {TFT_MIN_SEQS}; "
                    f"{dense_frac:.0%} hourly over {span_days}d span")
        report["tft"] = {"trained": False, "reason": tft_note}
        print("TFT:", tft_note)
    elif not torch_available:
        tft_note = "deferred: torch not installed (local/full image only)"
        report["tft"] = {"trained": False, "reason": tft_note}
        print("TFT:", tft_note)
    else:
        report["tft"] = await train_tft_gated(seqs)
        print("TFT:", report["tft"].get("verdict", report["tft"]))

    wx, wx_fetched, wx_cov = None, False, 0.0
    if use_weather:
        import json as _json
        from backend.app.config import DATA_DIR as _DD
        all_ts = [p["timestamp"] for pts in series.values() for p in pts]
        stations = _json.loads((_DD / "stations.json").read_text())["stations"]
        wx, wx_fetched = await ensure_weather_cache(
            stations, min(all_ts), max(all_ts), weather_cache)
    fires = load_fires(str(db))
    print(f"Fire history: {len(fires)} hotspots")
    samples, (wx_cov, fire_cov) = await build_samples(series, wx, fires)
    print(f"Training samples: {len(samples)} (need >={min_samples})"
          + (f" | weather: {wx_cov:.0%} "
             f"({'fetched' if wx_fetched else 'cache'})" if use_weather else "")
          + f" | fire-active: {fire_cov:.0%}")
    report["samples"] = len(samples)
    report["fires"] = {"hotspots": len(fires),
                       "sample_coverage": round(fire_cov, 3)}
    if use_weather:
        report["weather"] = {"coverage": round(wx_cov, 3),
                             "fetched_now": wx_fetched,
                             "cache": weather_cache}

    if len(samples) < min_samples:
        report["verdict"] = (
            f"REFUSED: {len(samples)} samples < {min_samples} minimum. "
            f"Keep the server running (hourly refresh) or deploy it, then re-run. "
            f"Rough target: 3-4 weeks of hourly data across 53 stations."
        )
        print("\n" + report["verdict"])
        return report

    split = int(len(samples) * 0.8)
    tr, te = samples[:split], samples[split:]
    Xtr = [f for _, f, _ in tr]
    ytr = [y for _, _, y in tr]
    Xte = [f for _, f, _ in te]
    yte = [y for _, _, y in te]
    persist = [f.get("pm25_now", 0.0) for f in Xte]
    p_err = [p - t for p, t in zip(persist, yte)]
    p_stats = stats(p_err)
    print(f"Holdout: train={len(tr)} test={len(te)} | "
          f"persistence MAE={p_stats['mae']} RMSE={p_stats['rmse']}")
    report["holdout"] = {"train_n": len(tr), "test_n": len(te),
                         "persistence": p_stats}

    if len(te) < MIN_TEST_ROWS:
        report["verdict"] = (f"REFUSED: only {len(te)} test rows "
                             f"(need >={MIN_TEST_ROWS}).")
        print("\n" + report["verdict"])
        return report

    saved = []

    async def train_and_gate(name, forecaster, path):
        rep = await forecaster.train(Xtr, ytr)
        if not rep.get("trained"):
            return {**rep, "saved": False}
        import numpy as np
        cols = forecaster._feature_names
        preds = []
        for f in Xte:
            X = np.array([[float(f.get(c, 0.0)) for c in cols]])
            preds.append(float(forecaster._model.predict(X)[0]))
        errs = [p - t for p, t in zip(preds, yte)]
        m_stats = stats(errs)
        r = round(pearson(preds, yte), 3)
        beats = m_stats["mae"] < p_stats["mae"] and r > MIN_R
        out_rep = {"trained": True, "backend": rep.get("backend"),
                   "model_mae": m_stats["mae"], "model_rmse": m_stats["rmse"],
                   "pearson_r": r, "saved": False}
        if beats:
            import joblib
            MODELS = PROJECT_ROOT / "backend" / "models"
            MODELS.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": forecaster._model, "features": cols,
                         **({"backend": forecaster._backend}
                            if hasattr(forecaster, "_backend") else {})},
                        str(MODELS / path))
            out_rep["saved"] = True
            out_rep["path"] = f"backend/models/{path}"
            saved.append(path)
            print(f"{name}: MAE {m_stats['mae']} < persist {p_stats['mae']}, "
                  f"r={r} -> SAVED to backend/models/{path}")
        else:
            print(f"{name}: MAE {m_stats['mae']} vs persist {p_stats['mae']}, "
                  f"r={r} -> NOT saved (fails guardrail)")
        return out_rep

    # Train WITHOUT a model_path so train() cannot persist before gating
    report["lgbm"] = await train_and_gate(
        "LGBM", LightGBMForecaster(model_path=None), "lgbm_pm25.joblib")
    if xgb_available:
        report["xgboost"] = await train_and_gate(
            "XGB", XGBoostForecaster(model_path=None), "xgboost_pm25.joblib")
    else:
        report["xgboost"] = {"trained": False,
                             "reason": "xgboost not installed locally",
                             "saved": False}

    if saved:
        report["verdict"] = (f"SAVED {saved}. Restart the server — the "
                             f"dashboard model badge flips from "
                             f"'statistical baseline' to the trained members, "
                             f"and /api/v1/accuracy/summary should improve. "
                             f"Commit backend/models/*.joblib to share them.")
    else:
        report["verdict"] = ("Models trained but did not beat persistence on "
                             "the holdout — nothing saved. More history needed.")
    print("\n" + report["verdict"])
    return report


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/aqi_data.db")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    ap.add_argument("--force-sklearn", action="store_true",
                    help="use sklearn HistGBM so weights load on Render free-tier")
    ap.add_argument("--no-weather", action="store_true",
                    help="ablation: train without ERA5 weather (proves its value)")
    args = ap.parse_args()
    report = await run_training(db=args.db, min_samples=args.min_samples,
                                force_sklearn=args.force_sklearn,
                                use_weather=not args.no_weather)
    out = PROJECT_ROOT / "scripts" / "training_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report: {out}")


if __name__ == "__main__":
    asyncio.run(main())
