"""
run_pipeline.py — ParkIQ end to end.

    python run_pipeline.py

Reads config.py, runs: load -> features -> hotspots -> CIS -> forecast ->
optimize -> dashboard, and writes everything to outputs/.

If config.DATA_PATH is missing it auto-generates the synthetic dataset, so this
ALWAYS runs and produces a full demo — then you point DATA_PATH at the real file.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd

import config
from src import loader, features, cis, forecast, optimizer, dashboard

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def banner(t):
    print("\n" + "=" * 68 + f"\n  {t}\n" + "=" * 68)


def ensure_data():
    # 1) the explicitly configured path
    if os.path.exists(config.DATA_PATH):
        return config.DATA_PATH
    # 2) auto-detect ANY real dataset dropped into data/ (csv/xlsb/xlsx/parquet),
    #    so judges can use the original file under any name/format with no edits.
    import glob
    cands = []
    for ext in ("*.xlsb", "*.xlsx", "*.csv", "*.parquet"):
        cands += glob.glob(os.path.join("data", ext))
    cands = [c for c in cands if "synthetic" not in os.path.basename(c).lower()]
    if cands:
        chosen = max(cands, key=os.path.getsize)   # the real file is the big one
        print(f"[run] Using dataset auto-detected in data/: {chosen}")
        return chosen
    # 3) fallback: synthetic demo data so the pipeline always runs
    print(f"[run] No dataset in data/ — generating synthetic demo data "
          f"(results will be illustrative, not the real Bengaluru numbers).")
    import synthetic_data
    os.makedirs("data", exist_ok=True)
    path = "data/synthetic_violations.csv"
    if not os.path.exists(path):
        synthetic_data.generate().to_csv(path, index=False)
    return path


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = ensure_data()

    banner("1/6  LOAD + SCHEMA DETECTION + PARKING FILTER")
    df, schema = loader.load(path, config.COLUMN_MAP, config.PARKING_KEYWORDS,
                             local_tz=getattr(config, "LOCAL_TZ", None))

    banner("2/6  FEATURES + HOTSPOT MINING")
    df = features.add_temporal(df, config.RUSH_HOURS)
    df, peak_hours = features.apply_peak_hours(
        df, mode=getattr(config, "PEAK_MODE", "data_driven"),
        fixed_rush=config.RUSH_HOURS)
    print(f"Peak hours used for 'peakedness' ({getattr(config,'PEAK_MODE','data_driven')}): "
          f"{peak_hours}")
    df = features.add_offence_severity(df, schema, config.OFFENCE_SEVERITY,
                                       config.OFFENCE_SEVERITY_DEFAULT)
    df, meta, mode = features.cluster_hotspots(
        df, schema,
        method=getattr(config, "SPATIAL_METHOD", "auto"),
        grid_m=getattr(config, "GRID_SIZE_M", 150),
        min_cluster_size=config.MIN_CLUSTER_SIZE,
        radius_m=config.HOTSPOT_RADIUS_M,
        hdbscan_max=getattr(config, "SPATIAL_HDBSCAN_MAX", 60000))
    print(f"Spatial mode: {mode} | method: {getattr(config,'SPATIAL_METHOD','auto')} "
          f"| hotspots found: {len(meta)}")

    banner("3/6  CONGESTION IMPACT SCORE (CIS)")
    cis_table = cis.compute_cis(df, meta, mode, config.CIS_WEIGHTS,
                                config.RUSH_HOURS, config.CRITICAL_KEYWORDS)
    print(cis_table.head(15).to_string(index=False))
    cis_path = os.path.join(config.OUTPUT_DIR, "cis_ranked_hotspots.csv")
    cis_table.to_csv(cis_path, index=False)

    banner("4/6  PROACTIVE FORECAST (LightGBM, time-split)")
    panel = forecast.add_features(forecast.build_panel(df, config.FORECAST_TIME_BIN))
    model, metrics, importances, fc_cols = forecast.train_forecast(
        panel, config.TEST_FRACTION, config.TOP_K)
    print("Validation metrics (held-out FUTURE):")
    for k, v in metrics.items():
        print(f"   {k:<18}: {v}")
    print("\nTop feature importances:")
    print(importances.head(8).to_string())
    next_fc = forecast.forecast_next(model, panel, fc_cols)

    banner("5/6  ENFORCEMENT OPTIMISER (patrol schedule)")
    name_map = dict(zip(cis_table["hotspot"], cis_table["name"]))
    mode_opt = getattr(config, "OPTIMIZE_MODE", "impact")
    if mode_opt == "coverage":
        target = getattr(config, "COVERAGE_TARGET", 0.80)
        total_v = len(df)   # ALL parking violations (honest denominator)
        schedule, achieved, n_hs = optimizer.coverage_deployment(
            cis_table, target, config.PATROL_UNITS, config.SHIFTS,
            total_violations=total_v)
        total_impact = round(float(schedule["impact"].sum()), 3)
        print(f"Coverage target: {target:.0%} of ALL {total_v:,} violations | "
              f"hotspots patrolled: {n_hs} | achieved: {achieved:.1%}")
        print(f"Patrols: {config.PATROL_UNITS} zones x {len(config.SHIFTS)} shifts | "
              f"avg stops/patrol/shift: {len(schedule)/(config.PATROL_UNITS*len(config.SHIFTS)):.1f}")
        # coverage curve (cumulative coverage vs #hotspots, over ALL violations)
        bv = cis_table.sort_values("violations", ascending=False).reset_index(drop=True)
        curve = pd.DataFrame({
            "n_hotspots": np.arange(1, len(bv) + 1),
            "coverage_pct": (bv["violations"].cumsum() / total_v * 100).round(2)})
        curve.to_csv(os.path.join(config.OUTPUT_DIR, "coverage_curve.csv"), index=False)
        print(curve.iloc[[0, n_hs-1, -1]].to_string(index=False))
    else:
        impact_df = optimizer.build_impact(df, cis_table, next_fc,
                                           config.SHIFTS, config.SHIFT_HOURS)
        schedule, total_impact = optimizer.optimize(
            impact_df, config.PATROL_UNITS, config.SHIFTS, name_map)
        print(f"Patrol units/shift: {config.PATROL_UNITS} | "
              f"total expected impact covered: {total_impact}")
        print(schedule.to_string(index=False))
    sched_path = os.path.join(config.OUTPUT_DIR, "patrol_schedule.csv")
    schedule.to_csv(sched_path, index=False)

    # ---- prescriptive routes (visiting order per shift) ----
    from src import routing
    coords = {h: (la, lo) for h, la, lo in
              zip(cis_table["hotspot"], cis_table["lat"], cis_table["lon"])}
    depot = getattr(config, "DEPOT_LATLON", None)
    routes_df, route_summary = (pd.DataFrame(), pd.DataFrame())
    if not schedule.empty and "hotspot" in schedule:
        routes_df, route_summary = routing.route_shifts(schedule, coords, depot)
        if not route_summary.empty:
            print("\nPatrol routes (TSP per shift):")
            print(route_summary.to_string(index=False))
            routes_df.to_csv(os.path.join(config.OUTPUT_DIR, "patrol_routes.csv"),
                             index=False)
            route_summary.to_csv(
                os.path.join(config.OUTPUT_DIR, "route_summary.csv"), index=False)

    banner("6/6  DASHBOARD")
    # small CSVs so the Streamlit app never needs the 26MB raw file at demo time
    (df.groupby(["dow", "hour"]).size().reset_index(name="count")
       .to_csv(os.path.join(config.OUTPUT_DIR, "temporal_profile.csv"), index=False))
    # equity: violations per (police division, hotspot) — incl '-1' (uncoverable),
    # so the app can compute coverage PER DIVISION for any patrolled set.
    zcol = schema.get("zone")
    if zcol:
        zser = df[zcol].astype(str).rename("zone")
        hser = df["hotspot"].astype(str).rename("hotspot")
        (df.groupby([zser, hser]).size().reset_index(name="n")
           .to_csv(os.path.join(config.OUTPUT_DIR, "equity_zone_hotspot.csv"),
                   index=False))
    # Kepler.gl time-lapse export: hotspot points by MONTH (real temporal signal;
    # do NOT animate by hour — that reflects challan-creation bias, not demand).
    kt = df[df["hotspot"].astype(str) != "-1"].copy()
    if "_lat" in kt.columns and len(kt):
        kt["month"] = kt["ts"].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
        (kt.groupby(["hotspot", "month"])
           .agg(lat=("_lat", "mean"), lon=("_lon", "mean"),
                violations=("hotspot", "size")).reset_index()
           .to_csv(os.path.join(config.OUTPUT_DIR, "kepler_timelapse.csv"), index=False))
    map_path = dashboard.build_map(
        df, cis_table, mode, os.path.join(config.OUTPUT_DIR, "hotspot_map.html"),
        routes_df=routes_df)
    chart_path = dashboard.build_charts(
        df, cis_table, os.path.join(config.OUTPUT_DIR, "charts.html"))

    # machine-readable summary
    summary = {
        "rows_analysed": int(len(df)),
        "total_parking_violations": int(len(df)),
        "spatial_mode": mode,
        "n_hotspots": int(len(meta)),
        "optimize_mode": mode_opt,
        "forecast_metrics": metrics,
        "total_expected_impact_covered": total_impact,
        "outputs": {
            "cis_csv": cis_path, "schedule_csv": sched_path,
            "routes_csv": os.path.join(config.OUTPUT_DIR, "patrol_routes.csv"),
            "map_html": map_path, "charts_html": chart_path,
        },
        "fleet_route_km": (round(float(route_summary["route_km"].sum()), 1)
                           if not route_summary.empty else 0.0),
    }
    if mode_opt == "coverage":
        summary["coverage_target"] = target
        summary["coverage_achieved"] = round(achieved, 4)
        summary["hotspots_patrolled"] = int(n_hs)
    with open(os.path.join(config.OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    banner("DONE")
    print("All artefacts written to ./outputs/")
    for k, v in summary["outputs"].items():
        print(f"   {k:<12}: {v}")


if __name__ == "__main__":
    main()
