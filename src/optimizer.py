"""
optimizer.py — turns the forecast + CIS into a deployable patrol schedule.

This is the "ACT" layer. Analytics that don't change a patrol roster don't win.

Expected impact of putting a patrol at hotspot h during shift s:
    impact[h,s] = predicted_next_count[h] * shift_share[h,s] * (CIS[h] / 100)
i.e. how many violations we expect there next period, weighted by how bad that
spot is for traffic flow (CIS), distributed across the day by that spot's own
historical hourly profile.

We then solve an integer program:
    maximise   sum impact[h,s] * x[h,s]
    s.t.       sum_h x[h,s] <= PATROL_UNITS        for every shift s
               sum_s x[h,s] <= MAX_SHIFTS_PER_SPOT for every hotspot h  (spread)
               x in {0,1}
Solved with PuLP (CBC). Falls back to a greedy solver if PuLP is unavailable.
"""
import numpy as np
import pandas as pd

MAX_SHIFTS_PER_SPOT = 2   # don't sink every patrol into one corner


def coverage_deployment(cis, target, patrol_units, shifts, rank_by="violations",
                        total_violations=None):
    """Select the MINIMUM set of hotspots whose violations reach `target` coverage
    (measured against ALL parking violations when total_violations is given), then
    split them into balanced, geographically-clustered patrol routes.

    Models reality: a patrol drives a multi-stop circuit per shift, so coverage is
    every junction on its route — not one spot per patrol. Returns
    (schedule_df, achieved_coverage, n_hotspots)."""
    c = cis.dropna(subset=["lat", "lon"]).copy()
    c = c.sort_values(rank_by, ascending=False).reset_index(drop=True)
    total = total_violations if total_violations else c["violations"].sum()
    c["cum"] = c["violations"].cumsum() / max(total, 1)
    n = int((c["cum"] < target).sum()) + 1
    n = min(n, len(c))
    S = c.head(n).copy()
    achieved = float(S["violations"].sum() / max(total, 1))

    # geographic zones -> one patrol owns a contiguous zone (short routes)
    from sklearn.cluster import KMeans
    k = int(min(patrol_units, len(S)))
    if k >= 2:
        S["patrol"] = KMeans(n_clusters=k, n_init=10,
                             random_state=0).fit_predict(S[["lat", "lon"]].values)
    else:
        S["patrol"] = 0

    # within each patrol zone, spread stops across shifts (round-robin by volume)
    rows = []
    for p, grp in S.groupby("patrol"):
        grp = grp.sort_values("violations", ascending=False)
        for i, (_, r) in enumerate(grp.iterrows()):
            rows.append({"hotspot": r["hotspot"], "name": r["name"],
                         "lat": r["lat"], "lon": r["lon"], "patrol": int(p),
                         "shift": shifts[i % len(shifts)],
                         "violations": int(r["violations"]),
                         "impact": round(float(r["CIS"]) / 10, 2)})
    return pd.DataFrame(rows), achieved, n


def _shift_shares(df, shifts, shift_hours):
    """Per-hotspot fraction of violations falling in each shift."""
    work = df[df["hotspot"].astype(str) != "-1"].copy()
    hour_to_shift = {}
    for s in shifts:
        for h in shift_hours[s]:
            hour_to_shift[h] = s
    work["shift"] = work["hour"].map(hour_to_shift)
    counts = work.groupby(["hotspot", "shift"]).size().rename("n").reset_index()
    totals = counts.groupby("hotspot")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    return counts.pivot(index="hotspot", columns="shift", values="share").fillna(0)


def build_impact(df, cis_table, forecast_next, shifts, shift_hours):
    shares = _shift_shares(df, shifts, shift_hours)
    cis = cis_table.set_index("hotspot")["CIS"]
    fc = forecast_next.set_index("hotspot")["pred_next"]
    rows = []
    for h in shares.index:
        if h not in cis.index or h not in fc.index:
            continue
        for s in shifts:
            share = shares.loc[h, s] if s in shares.columns else 0.0
            impact = float(fc[h]) * float(share) * float(cis[h]) / 100.0
            rows.append({"hotspot": h, "shift": s, "impact": impact})
    return pd.DataFrame(rows)


def optimize(impact_df, patrol_units, shifts, name_map=None,
             max_shifts_per_spot=MAX_SHIFTS_PER_SPOT):
    try:
        return _optimize_ilp(impact_df, patrol_units, shifts, name_map,
                             max_shifts_per_spot)
    except Exception as e:  # pragma: no cover
        print(f"[optimizer] ILP unavailable ({e}); using greedy fallback.")
        return _optimize_greedy(impact_df, patrol_units, shifts, name_map)


def _optimize_ilp(impact_df, patrol_units, shifts, name_map, max_per_spot):
    import pulp
    prob = pulp.LpProblem("patrol_allocation", pulp.LpMaximize)
    x = {}
    for _, r in impact_df.iterrows():
        x[(r["hotspot"], r["shift"])] = pulp.LpVariable(
            f"x_{r['hotspot']}_{r['shift']}", cat="Binary")
    # objective
    prob += pulp.lpSum(r["impact"] * x[(r["hotspot"], r["shift"])]
                       for _, r in impact_df.iterrows())
    # patrol capacity per shift
    for s in shifts:
        prob += pulp.lpSum(x[(h, ss)] for (h, ss) in x if ss == s) <= patrol_units
    # spread per hotspot
    for h in impact_df["hotspot"].unique():
        prob += pulp.lpSum(x[(hh, ss)] for (hh, ss) in x if hh == h) <= max_per_spot
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    chosen = [{"hotspot": h, "shift": s,
               "impact": float(impact_df[(impact_df["hotspot"] == h) &
                                         (impact_df["shift"] == s)]["impact"].iloc[0])}
              for (h, s), var in x.items() if var.value() == 1]
    return _format(pd.DataFrame(chosen), name_map, pulp.value(prob.objective))


def _optimize_greedy(impact_df, patrol_units, shifts, name_map):
    cap = {s: patrol_units for s in shifts}
    used = {}
    chosen = []
    for _, r in impact_df.sort_values("impact", ascending=False).iterrows():
        sh, ht, im = r["shift"], r["hotspot"], r["impact"]
        if cap.get(sh, 0) <= 0:
            continue
        if used.get(ht, 0) >= MAX_SHIFTS_PER_SPOT:
            continue
        chosen.append(dict(hotspot=ht, shift=sh, impact=im))
        cap[sh] -= 1
        used[ht] = used.get(ht, 0) + 1
    return _format(pd.DataFrame(chosen), name_map,
                   sum(c["impact"] for c in chosen))


def _format(sched, name_map, total):
    if sched.empty:
        return sched, 0.0
    if name_map is not None:
        sched["name"] = sched["hotspot"].map(name_map)
    sched = sched.sort_values(["shift", "impact"], ascending=[True, False])
    sched["impact"] = sched["impact"].round(3)
    return sched.reset_index(drop=True), round(float(total), 3)
