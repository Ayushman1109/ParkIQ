"""
routing.py — prescriptive patrol ROUTES (the order to visit, not just the set).

The optimiser decides *which* hotspots each shift covers. This decides the
*sequence*: for each shift we solve a small Travelling-Salesman route over the
assigned junctions (nearest-neighbour construction + 2-opt improvement) using
great-circle distances, so a patrol drives the shortest loop hitting every
high-impact spot. Fast, deterministic, and far simpler than an RL agent while
giving the same operational payoff.
"""
import numpy as np
import pandas as pd


def _haversine_km(a, b):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def _dist_matrix(pts):
    n = len(pts)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _haversine_km(pts[i], pts[j])
    return D


def _nearest_neighbour(D, start=0):
    n = len(D)
    unvisited = set(range(n))
    tour = [start]
    unvisited.discard(start)
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: D[last, j])
        tour.append(nxt)
        unvisited.discard(nxt)
    return tour


def _path_len(tour, D):
    return sum(D[tour[i], tour[i + 1]] for i in range(len(tour) - 1))


def _two_opt(tour, D):
    """2-opt for an OPEN path with a fixed start node (no return edge counted)."""
    best = tour[:]
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = best[i - 1], best[i]
                c = best[k]
                d = best[k + 1] if k + 1 < n else None
                delta = (D[a, c] - D[a, b])
                if d is not None:
                    delta += (D[b, d] - D[c, d])
                if delta < -1e-9:
                    best[i:k + 1] = best[i:k + 1][::-1]
                    improved = True
    return best


def _solve_open_tsp(D):
    """Best open path from node 0: 2-opt seeded from BOTH a nearest-neighbour
    tour and the identity (input) order; keep the shorter. This guarantees the
    result is no worse than the naive input ordering."""
    nn = _two_opt(_nearest_neighbour(D, start=0), D)
    ident = _two_opt(list(range(len(D))), D)
    return nn if _path_len(nn, D) <= _path_len(ident, D) else ident


def route_shifts(schedule, coords, depot=None):
    """schedule: df with hotspot, shift, impact, name (+ optional patrol).
       coords: dict hotspot -> (lat, lon).
       depot: (lat,lon) start/end, or None to start at the top-impact stop.
    Routes per (patrol, shift) if a 'patrol' column exists, else per shift.
    Returns (routes_df, per_group_summary_df)."""
    rows, summary = [], []
    has_patrol = "patrol" in schedule.columns
    keys = ["patrol", "shift"] if has_patrol else ["shift"]
    for key, grp in schedule.groupby(keys):
        patrol = key[0] if has_patrol else None
        shift = key[1] if has_patrol else key
        grp = grp.dropna(subset=["hotspot"]).copy()
        pts, names, hs = [], [], []
        for _, r in grp.sort_values("impact", ascending=False).iterrows():
            c = coords.get(r["hotspot"])
            if c is None or pd.isna(c[0]) or pd.isna(c[1]):
                continue
            pts.append(c); names.append(r.get("name", r["hotspot"])); hs.append(r["hotspot"])
        if len(pts) == 0:
            continue
        use_depot = depot is not None
        nodes = ([tuple(depot)] + pts) if use_depot else pts
        labels = (["DEPOT"] + names) if use_depot else names
        ids = (["DEPOT"] + hs) if use_depot else hs
        D = _dist_matrix(nodes)
        tour = _solve_open_tsp(D)
        naive_km = _path_len(list(range(len(nodes))), D)
        cum = 0.0
        for order, idx in enumerate(tour):
            leg = 0.0 if order == 0 else D[tour[order - 1], idx]
            cum += leg
            rec = {"shift": shift, "order": order, "hotspot": ids[idx],
                   "name": labels[idx], "lat": nodes[idx][0], "lon": nodes[idx][1],
                   "leg_km": round(leg, 3), "cum_km": round(cum, 3)}
            if has_patrol:
                rec["patrol"] = patrol
            rows.append(rec)
        total = cum
        saved = 100 * (naive_km - total) / naive_km if naive_km > 0 else 0.0
        s = {"shift": shift, "stops": len(pts), "naive_km": round(naive_km, 2),
             "route_km": round(total, 2), "km_saved_pct": round(saved, 1)}
        if has_patrol:
            s["patrol"] = patrol
        summary.append(s)
    return pd.DataFrame(rows), pd.DataFrame(summary)
