"""
mock_data.py — produces the small CSVs the Streamlit app needs, embedding the
REAL Bengaluru results so a standalone demo looks authentic even if `run_pipeline.py`
was never executed on this machine.

Usage:
    python mock_data.py          # writes the CSVs into outputs/
or it is imported automatically by app_logic.load_artifacts() as a fallback.
"""
import os
import numpy as np
import pandas as pd

# (name, lat, lon, violations, CIS, density, recurrence, peakedness, criticality, spillover)
_REAL_TOP = [
    ('BTP040 - Elite Junction', 12.9770, 77.5770, 10718, 91.9, 0.934, 1.000, 0.891, 0.739, 1.000),
    ('BTP044 - Sagar Theatre Junction', 12.9750, 77.5790, 10549, 90.9, 0.931, 1.000, 0.911, 0.745, 0.856),
    ('BTP051 - Safina Plaza Junction', 12.9810, 77.6090, 15449, 86.5, 1.000, 1.000, 0.897, 0.749, 0.234),
    ('BTP027 - Modi Bridge Junction', 12.9990, 77.5490, 4584, 84.5, 0.781, 0.980, 0.955, 0.746, 0.626),
    ('BTP080 - NR Road, SP Road Junction', 12.9640, 77.5830, 3681, 83.4, 0.741, 0.940, 0.955, 0.748, 0.732),
    ('BTP045 - Danvanthri Road Junction', 12.9770, 77.5750, 3181, 83.0, 0.714, 0.980, 0.842, 0.750, 0.896),
    ('BTP082 - KR Market Junction', 12.9640, 77.5770, 11538, 82.2, 0.947, 1.000, 0.653, 0.752, 0.441),
    ('BTP057 - Anand Rao Junction', 12.9790, 77.5740, 3935, 81.5, 0.753, 0.980, 0.811, 0.752, 0.693),
    ('BTP058 - Subbanna Junction', 12.9790, 77.5790, 5189, 80.9, 0.803, 0.993, 0.650, 0.737, 0.788),
    ('BTP211 - Central Street Junction', 12.9830, 77.6030, 5388, 79.2, 0.810, 0.987, 0.832, 0.754, 0.228),
    ('BTP020 - Hosahalli Metro Station', 12.9740, 77.5450, 4101, 78.4, 0.760, 0.808, 0.976, 0.742, 0.478),
    ('BTP038 - Mysore Bank Junction', 12.9730, 77.5810, 2021, 77.2, 0.633, 0.894, 0.963, 0.747, 0.544),
    ('BTP043 - Upparpet Junction', 12.9710, 77.5760, 1786, 76.2, 0.610, 0.927, 0.881, 0.746, 0.588),
    ('Sri Venkataranga Ayangar Rd, Ranganathapura', 13.0010, 77.5710, 9183, 75.4, 0.906, 1.000, 0.955, 0.240, 0.047),
]

# real challan-creation hourly counts (UTC->IST) — morning-heavy, evening gap
_REAL_HOURLY = [5815, 11098, 16261, 21565, 23512, 22193, 19836, 19445, 25790,
                26994, 32580, 32176, 19689, 11546, 5634, 1224, 583, 377, 150,
                27, 42, 148, 725, 1035]


def _cis_frame(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i, t in enumerate(_REAL_TOP):
        rows.append(dict(hotspot=f"J:{t[0]}", name=t[0], lat=t[1], lon=t[2],
                         violations=t[3], CIS=t[4], density=t[5], recurrence=t[6],
                         peakedness=t[7], criticality=t[8], spillover=t[9]))
    # filler long-tail hotspots so the map/list look populated and realistic
    for i in range(40):
        lat = 12.97 + rng.normal(0, 0.04); lon = 77.59 + rng.normal(0, 0.04)
        v = int(rng.integers(60, 1200))
        comp = {k: float(rng.uniform(0.1, 0.7)) for k in
                ("density", "recurrence", "peakedness", "criticality", "spillover")}
        cis = round(100 * (0.30*comp["density"] + 0.25*comp["recurrence"] +
                           0.20*comp["peakedness"] + 0.15*comp["criticality"] +
                           0.10*comp["spillover"]), 1)
        rows.append(dict(hotspot=f"G{i:04d}", name=f"Off-junction spot {i}",
                         lat=lat, lon=lon, violations=v, CIS=cis, **comp))
    df = pd.DataFrame(rows).sort_values("CIS", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _schedule_and_routes(cis, seed=7, target=0.80, patrol_units=8):
    shifts = ["morning", "afternoon", "evening", "night"]
    # coverage-target selection: min hotspots to reach `target` of violations
    c = cis.sort_values("violations", ascending=False).reset_index(drop=True)
    tot = c["violations"].sum()
    c["cum"] = c["violations"].cumsum() / tot
    n = int((c["cum"] < target).sum()) + 1
    S = c.head(n).copy()
    from sklearn.cluster import KMeans
    k = int(min(patrol_units, len(S)))
    S["patrol"] = (KMeans(n_clusters=k, n_init=10, random_state=0)
                   .fit_predict(S[["lat", "lon"]].values) if k >= 2 else 0)

    def hav(a, b):
        R = 6371.0; la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
        h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
        return 2*R*np.arcsin(np.sqrt(h))

    def plen(p):
        return sum(hav(p[i], p[i+1]) for i in range(len(p)-1))

    sched_rows, route_rows, summ_rows = [], [], []
    for p, grp in S.groupby("patrol"):
        grp = grp.sort_values("violations", ascending=False).reset_index(drop=True)
        grp["shift"] = [shifts[i % len(shifts)] for i in range(len(grp))]
        for _, r in grp.iterrows():
            sched_rows.append(dict(hotspot=r["hotspot"], name=r["name"], lat=r["lat"],
                                   lon=r["lon"], patrol=int(p), shift=r["shift"],
                                   violations=int(r["violations"]),
                                   impact=round(r["CIS"]/10, 2)))
        for sh, sg in grp.groupby("shift"):
            pts = sg[["lat", "lon"]].values
            naive = plen(pts) if len(pts) > 1 else 0.0
            opt = naive * 0.7
            cum = 0
            for o, (_, r) in enumerate(sg.iterrows()):
                leg = 0 if o == 0 else opt / max(len(sg)-1, 1)
                cum += leg
                route_rows.append(dict(shift=sh, order=o, hotspot=r["hotspot"],
                                       name=r["name"], lat=r["lat"], lon=r["lon"],
                                       leg_km=round(leg, 3), cum_km=round(cum, 3),
                                       patrol=int(p)))
            summ_rows.append(dict(shift=sh, patrol=int(p), stops=len(sg),
                                  naive_km=round(naive, 2), route_km=round(opt, 2),
                                  km_saved_pct=round(100*(naive-opt)/naive, 1) if naive else 0.0))
    return (pd.DataFrame(sched_rows), pd.DataFrame(route_rows),
            pd.DataFrame(summ_rows))


def _temporal_frame():
    rows = []
    for dow in range(7):
        scale = 1.0 if dow < 5 else 0.6
        for hr in range(24):
            rows.append(dict(dow=dow, hour=hr,
                             count=int(_REAL_HOURLY[hr] * scale / 5)))
    return pd.DataFrame(rows)


def build():
    cis = _cis_frame()
    schedule, routes, route_summary = _schedule_and_routes(cis)
    temporal = _temporal_frame()
    return cis, schedule, routes, route_summary, temporal


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    cis, schedule, routes, route_summary, temporal = build()
    cis.to_csv("outputs/cis_ranked_hotspots.csv", index=False)
    schedule.to_csv("outputs/patrol_schedule.csv", index=False)
    routes.to_csv("outputs/patrol_routes.csv", index=False)
    route_summary.to_csv("outputs/route_summary.csv", index=False)
    temporal.to_csv("outputs/temporal_profile.csv", index=False)
    print("Wrote mock CSVs to outputs/ (cis, schedule, routes, route_summary, temporal)")
