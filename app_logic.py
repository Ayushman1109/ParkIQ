"""
app_logic.py — all the pure logic behind the Streamlit demo, kept UI-free so it
can be unit-tested and reused. app.py imports from here.

Design principles for a demo-survivable app:
  * Everything reads the small CSVs the pipeline emits (no 26MB raw file needed).
  * Numbers are NEVER hardcoded — the "highest-impact junction" and "morning route
    km" are looked up live, so the demo can't show a stale/wrong figure.
  * The NL dispatcher is deterministic keyword/intent matching, not a live LLM —
    zero hallucination risk on stage.
"""
import os
import re
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Data loading (with graceful fallback to mock data)
# --------------------------------------------------------------------------- #
def load_artifacts(outdir="outputs"):
    def _read(name):
        p = os.path.join(outdir, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    cis = _read("cis_ranked_hotspots.csv")
    schedule = _read("patrol_schedule.csv")
    routes = _read("patrol_routes.csv")
    route_summary = _read("route_summary.csv")
    temporal = _read("temporal_profile.csv")
    curve = _read("coverage_curve.csv")
    equity = _read("equity_zone_hotspot.csv")

    total_violations = None
    sp = os.path.join(outdir, "summary.json")
    if os.path.exists(sp):
        import json
        try:
            total_violations = json.load(open(sp)).get("total_parking_violations")
        except Exception:
            total_violations = None

    if cis is None:                       # demo fallback
        import mock_data
        cis, schedule, routes, route_summary, temporal = mock_data.build()
    cis = add_ranks(cis)
    if total_violations is None:
        total_violations = int(cis["violations"].sum())
    if curve is None:
        curve = coverage_curve(cis, total_violations)
    if equity is None:
        equity = _mock_equity(cis)
    return {"cis": cis, "schedule": schedule, "routes": routes,
            "route_summary": route_summary, "temporal": temporal,
            "curve": curve, "total_violations": total_violations,
            "equity": equity}


def _mock_equity(cis):
    """Synthesize a (zone, hotspot, n) table from a cis frame for the demo fallback:
    assign each hotspot to a pseudo-division, and add a few all-uncovered divisions."""
    rng = np.random.default_rng(3)
    zones = [f"Division {i+1}" for i in range(12)]
    rows = []
    for _, r in cis.iterrows():
        z = zones[int(rng.integers(0, 8))]            # hotspots cluster in 8 zones
        rows.append({"zone": z, "hotspot": r["hotspot"], "n": int(r["violations"])})
    # 4 outer divisions that have violations but no hotspots (uncovered)
    for z in zones[8:]:
        rows.append({"zone": z, "hotspot": "-1", "n": int(rng.integers(800, 3000))})
    return pd.DataFrame(rows)


def add_ranks(cis):
    cis = cis.copy()
    cis["raw_rank"] = cis["violations"].rank(ascending=False, method="min").astype(int)
    cis["diff_rank"] = cis["CIS"].rank(ascending=False, method="min").astype(int)
    cis["move"] = cis["raw_rank"] - cis["diff_rank"]   # +ve = promoted by diffusion
    # CONGESTION-IMPACT WEIGHT per violation: a car blocking a critical arterial in a
    # busy hour hurts flow far more than one on a quiet street off-peak. We derive a
    # 0.5x–2.0x multiplier from each hotspot's criticality & peakedness (consequence,
    # not volume — density is excluded to avoid double-counting), then impact_volume
    # = violations x multiplier. (Proxy for flow impact; we have no live speed feed.)
    for col in ("criticality", "peakedness"):
        if col not in cis.columns:
            cis[col] = 0.5
    sev = (cis["criticality"].fillna(0.5) + cis["peakedness"].fillna(0.5)) / 2
    cis["severity_x"] = (0.5 + 1.5 * sev).round(3)        # 0.5x .. 2.0x
    cis["impact_volume"] = (cis["violations"] * cis["severity_x"]).round(1)
    return cis


def _q_total(cis, quantity, total_violations):
    """Denominator for coverage: all violations (incl. sparse) for 'violations',
    else the sum of the chosen quantity over hotspots."""
    if quantity == "violations" and total_violations:
        return total_violations
    return float(cis[quantity].sum())


def coverage_curve(cis, total_violations, quantity="violations"):
    """Cumulative coverage (% of total `quantity`) as #hotspots increases."""
    bv = cis.sort_values(quantity, ascending=False).reset_index(drop=True)
    denom = _q_total(cis, quantity, total_violations)
    return pd.DataFrame({
        "n_hotspots": np.arange(1, len(bv) + 1),
        "coverage_pct": (bv[quantity].cumsum() / max(denom, 1) * 100)})


def knee_point(curve):
    """Point of max curvature (max distance to the chord) — the efficiency 'knee'
    beyond which each extra hotspot buys very little coverage."""
    x = curve["n_hotspots"].values.astype(float)
    y = curve["coverage_pct"].values.astype(float)
    if len(x) < 3:
        return int(x[-1]), float(y[-1])
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0) or 1.0
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    i = int(np.argmax(dist))
    return int(x[i]), float(y[i])


def select_for_target(cis, target_pct, total_violations, quantity="violations"):
    """How many hotspots reach target_pct of `quantity`, and the set."""
    bv = cis.sort_values(quantity, ascending=False).reset_index(drop=True)
    denom = _q_total(cis, quantity, total_violations)
    cum = bv[quantity].cumsum() / max(denom, 1) * 100
    n = int((cum < target_pct).sum()) + 1
    n = min(n, len(bv))
    chosen = set(bv.head(n)["hotspot"])
    return {"n": n, "coverage": float(cum.iloc[n-1]), "hotspots": chosen}


# --------------------------------------------------------------------------- #
# Equity — coverage fairness across police divisions
# --------------------------------------------------------------------------- #
def equity_by_zone(equity, patrolled):
    """Per-division coverage for a given patrolled hotspot set."""
    eq = equity.copy()
    total = eq.groupby("zone")["n"].sum().rename("total")
    cov = (eq[eq["hotspot"].astype(str).isin(set(map(str, patrolled)))]
           .groupby("zone")["n"].sum().rename("covered"))
    out = pd.concat([total, cov], axis=1).fillna(0.0)
    out["coverage_pct"] = (100 * out["covered"] / out["total"]).round(1)
    return out.sort_values("coverage_pct").reset_index()


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def equity_summary(zone_df):
    cov = zone_df["coverage_pct"].values
    return {"n_zones": len(zone_df),
            "below_50": int((cov < 50).sum()),
            "zero": int((cov == 0).sum()),
            "min": float(np.min(cov)) if len(cov) else 0.0,
            "median": float(np.median(cov)) if len(cov) else 0.0,
            "gap": float(np.max(cov) - np.min(cov)) if len(cov) else 0.0,
            "gini": round(_gini(cov), 3)}


def division_top_hotspots(equity):
    """Best (highest-volume, non-sparse) hotspot per division — the fairness floor."""
    eq = equity[equity["hotspot"].astype(str) != "-1"]
    if eq.empty:
        return set()
    idx = eq.groupby("zone")["n"].idxmax()
    return set(eq.loc[idx, "hotspot"].astype(str))


def coverage_of(cis, patrolled, total_violations):
    sub = cis[cis["hotspot"].astype(str).isin(set(map(str, patrolled)))]
    return round(100 * sub["violations"].sum() / max(total_violations, 1), 1)


def greedy_zone_cover(cis, total_violations, radius_m, target_pct, quantity="violations"):
    """Greedy Maximum-Coverage with a 'zone of influence' radius: each chosen centre
    absorbs every hotspot within radius_m (one patrol beat). Iteratively pick the
    centre capturing the most *uncovered* `quantity` until target_pct is reached.
    Larger radius -> fewer zones but bigger beats. Returns n_zones, coverage, centres."""
    c = cis.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    lat = np.radians(c["lat"].values); lon = np.radians(c["lon"].values)
    v = c[quantity].values.astype(float)
    R = 6_371_000.0
    dlat = lat[:, None] - lat[None, :]; dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat)[:, None]*np.cos(lat)[None, :]*np.sin(dlon/2)**2
    D = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    within = D <= radius_m
    covered = np.zeros(len(c), dtype=bool)
    centres, got = [], 0.0
    denom = _q_total(cis, quantity, total_violations)
    tgt = target_pct / 100 * denom
    while got < tgt and not covered.all():
        gain = np.where(within & ~covered[None, :], v[None, :], 0).sum(axis=1)
        i = int(np.argmax(gain))
        if gain[i] <= 0:
            break
        newly = within[i] & ~covered
        covered |= newly; got += v[newly].sum(); centres.append(c["hotspot"].iloc[i])
    return {"n_zones": len(centres), "coverage": round(100*got/max(denom, 1), 1),
            "centres": centres}


SHIFT_WORDS = ["morning", "afternoon", "evening", "night"]


def hotspot_division(equity):
    """Map each hotspot -> its dominant police division."""
    if equity is None:
        return {}
    eq = equity[equity["hotspot"].astype(str) != "-1"]
    if eq.empty:
        return {}
    top = eq.sort_values("n", ascending=False).drop_duplicates("hotspot")
    return dict(zip(top["hotspot"].astype(str), top["zone"].astype(str)))


def _haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(h))


def tsp_route(coords):
    """Nearest-neighbour + 2-opt open path. coords: list of (lat,lon). Returns
    (order indices, total_km)."""
    n = len(coords)
    if n <= 1:
        return list(range(n)), 0.0
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = D[j, i] = _haversine(coords[i], coords[j])
    # nearest neighbour
    unvis = set(range(1, n)); tour = [0]
    while unvis:
        last = tour[-1]; nxt = min(unvis, key=lambda j: D[last, j])
        tour.append(nxt); unvis.discard(nxt)
    # 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(1, n-1):
            for k in range(i+1, n):
                a, b = tour[i-1], tour[i]; c = tour[k]
                d = tour[k+1] if k+1 < n else None
                delta = D[a, c] - D[a, b] + (D[b, d]-D[c, d] if d is not None else 0)
                if delta < -1e-9:
                    tour[i:k+1] = tour[i:k+1][::-1]; improved = True
    total = sum(D[tour[i], tour[i+1]] for i in range(n-1))
    return tour, round(total, 1)


def route_over(df_sub):
    """Build a pydeck-ready route over a set of hotspots (df with lat,lon,name)."""
    df_sub = df_sub.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if len(df_sub) == 0:
        return None
    coords = list(zip(df_sub["lat"], df_sub["lon"]))
    order, km = tsp_route(coords)
    ordered = df_sub.iloc[order]
    return {"paths": [{"path": [[float(lo), float(la)]
                                for la, lo in zip(ordered["lat"], ordered["lon"])]}],
            "stops": [{"order": i, "name": r["name"], "lat": r["lat"], "lon": r["lon"]}
                      for i, (_, r) in enumerate(ordered.iterrows())],
            "km": km}


# --------------------------------------------------------------------------- #
# Business case — fuel, money, hours
# --------------------------------------------------------------------------- #
def business_case(naive_km, opt_km, n_vehicles=8, km_per_litre=8.0,
                  fuel_price=95.0, city_speed_kmh=20.0, days=30):
    """Translate routing savings into operational savings. All assumptions explicit
    and editable in the UI. Per-day figures are for the whole patrolled fleet."""
    km_saved_day = max(naive_km - opt_km, 0.0)
    litres_day = km_saved_day / max(km_per_litre, 1e-9)
    rupees_day = litres_day * fuel_price
    hours_day = km_saved_day / max(city_speed_kmh, 1e-9)   # driving time avoided
    return {"km_saved_day": round(km_saved_day, 1),
            "litres_day": round(litres_day, 1),
            "rupees_day": round(rupees_day, 0),
            "rupees_month": round(rupees_day * days, 0),
            "rupees_year": round(rupees_day * 365, 0),
            "hours_day": round(hours_day, 1),
            "hours_month": round(hours_day * days, 0)}


# --------------------------------------------------------------------------- #
# What-if disruption simulation
# --------------------------------------------------------------------------- #
def simulate_disruption(cis, hotspot_id, impact_multiplier, target_pct,
                        total_violations, quantity="impact_volume"):
    """Simulate a disruption (metro works / road closure / event) that worsens
    congestion at a hotspot: scale its impact, re-rank, re-select for the target,
    and report whether it enters the patrol set + the rank shift. (Re-routing here
    means re-selection + re-sequencing on straight-line distance, not road-graph
    avoidance — we have no road network.)"""
    base = add_ranks(cis)
    sim = base.copy()
    sim["violations"] = sim["violations"].astype(float)
    sim["CIS"] = sim["CIS"].astype(float)
    mask = sim["hotspot"].astype(str) == str(hotspot_id)
    if not mask.any():
        return None
    before_rank = int(base.loc[mask, "diff_rank"].iloc[0])
    sim.loc[mask, "violations"] = sim.loc[mask, "violations"] * impact_multiplier
    sim.loc[mask, "CIS"] = np.minimum(sim.loc[mask, "CIS"] * impact_multiplier, 100)
    sim = add_ranks(sim)
    after_rank = int(sim.loc[mask, "diff_rank"].iloc[0])
    sel_before = select_for_target(base, target_pct, total_violations, quantity)
    sel_after = select_for_target(sim, target_pct, total_violations, quantity)
    name = base.loc[mask, "name"].iloc[0]
    return {"name": name, "before_rank": before_rank, "after_rank": after_rank,
            "in_set_before": str(hotspot_id) in sel_before["hotspots"],
            "in_set_after": str(hotspot_id) in sel_after["hotspots"],
            "n_before": sel_before["n"], "n_after": sel_after["n"],
            "sim_cis": float(sim.loc[mask, "CIS"].iloc[0])}


def movers(cis, n=5):
    cols = ["name", "violations", "CIS", "raw_rank", "diff_rank", "move", "spillover"]
    promoted = cis.sort_values("move", ascending=False).head(n)[cols]
    demoted = cis.sort_values("move").head(n)[cols]
    return promoted, demoted


# --------------------------------------------------------------------------- #
# Module 1 — deterministic NL dispatcher
# --------------------------------------------------------------------------- #
SHIFT_WORDS = ["morning", "afternoon", "evening", "night"]


def nl_dispatch(query, cis, routes, hot_div=None):
    """Deterministic NL dispatcher (no LLM, no network). Robust to free-form English:
    synonyms, number-words, fuzzy division/junction matching, and a smart fallback
    that always returns something useful rather than a dead end."""
    import difflib
    q = (query or "").strip().lower()
    if not q:
        return {"intent": "idle", "message": "Ask me anything: e.g. \'where\'s the worst "
                "parking problem?\', \'plan a patrol for Koramangala\', \'top 5 chokepoints\'."}

    toks = re.findall(r"[a-z]+", q)
    tokset = set(toks)
    def has(words):
        return any(w in q for w in words)

    NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
           "nine":9,"ten":10,"twelve":12,"fifteen":15,"twenty":20,"couple":2,"few":3,
           "several":5,"handful":5}
    n = None
    mnum = re.search(r"\b(\d{1,3})\b", q)
    if mnum:
        n = int(mnum.group(1))
    else:
        for w, val in NUM.items():
            if w in tokset:
                n = val; break

    ROUTE = ["route","path","patrol","cover","visit","loop","circuit","drive","dispatch",
             "send","deploy","plan","tour","beat"]
    IMPACT = ["impact","worst","highest","biggest","critical","chokepoint","choke","priority",
              "congest","severe","hardest","problem","problems","bad","trouble","focus",
              "hotspot","hotspots","danger"]
    VOLUME = ["violation","violations","ticket","tickets","busiest","busy","volume","frequent",
              "offence","offences","challan","challans","count"]
    LISTW = ["list","rank","ranking","show","which","what","where","give","find","top"]
    wants_route = has(ROUTE)
    by_volume = has(VOLUME) and not has(IMPACT)
    rank_col = "violations" if by_volume else "CIS"
    metric_word = "violations" if by_volume else "congestion impact"

    c = cis.copy()
    if hot_div:
        c["division"] = c["hotspot"].astype(str).map(hot_div).fillna("\u2014")
    names = c["name"].astype(str)

    GENERIC = {"junction","road","main","cross","circle","gate","signal","metro","station",
               "market","layout","nagar","park","the","with","and","route","patrol","show",
               "near","that","this","cover","them","plan","for","top","most","high","what",
               "where","which","worst","best"}
    def match_division(divs):
        for d in sorted(divs, key=len, reverse=True):
            if d.lower() in q:
                return d
        for d in divs:
            for w in re.findall(r"[a-z]+", d.lower()):
                if len(w) >= 5 and w not in GENERIC:
                    if w in tokset or difflib.get_close_matches(w, toks, n=1, cutoff=0.84):
                        return d
        return None
    def match_junction():
        for nm in names:
            if nm.lower() in q:
                return nm
        for nm in names:
            for w in re.findall(r"[a-z]+", nm.lower()):
                if len(w) >= 5 and w not in GENERIC:
                    if w in tokset or difflib.get_close_matches(w, toks, n=1, cutoff=0.86):
                        return nm
        return None

    def match_area():
        """Return (df_of_matching_hotspots, label) for a BTP code or an area word
        that appears in one or more hotspot names (e.g. 'Koramangala')."""
        code = re.search(r"btp\s*0*(\d+)", q)
        if code:
            num = code.group(1)
            m = names.str.lower().str.contains(rf"btp0*{num}\b", regex=True)
            if m.any():
                return c[m], f"BTP{num}"
        for nm in names:
            if nm.lower() in q:
                return c[names == nm], nm
        for w in sorted(set(toks), key=len, reverse=True):
            if len(w) >= 5 and w not in GENERIC:
                m = names.str.lower().str.contains(re.escape(w))
                if m.any():
                    return c[m], w.title()
        return None, None

    divisions = sorted(set(hot_div.values())) if hot_div else []
    division = match_division(divisions) if divisions else None

    for sh in SHIFT_WORDS:
        if sh in q and wants_route:
            msg = f"Overlaying the {sh} patrol routes."
            if routes is not None and not routes.empty:
                sub = routes[routes["shift"] == sh]
                if len(sub):
                    if "patrol" in sub.columns:
                        msg = (f"Overlaying {sub['patrol'].nunique()} {sh} patrol routes \u2014 "
                               f"{len(sub)} stops, ~{sub.groupby('patrol')['cum_km'].max().sum():.0f} km total.")
                    else:
                        msg = (f"Overlaying the {sh} route \u2014 {sub['cum_km'].max():.1f} km, "
                               f"{len(sub)} stops.")
            return {"intent": "route", "shift": sh, "message": msg}

    if division is not None:
        cand = c[c["division"] == division].sort_values(rank_col, ascending=False)
        if not cand.empty:
            picked = cand.head(n or 5)
            res = {"intent": "division", "division": division,
                   "table": picked[["name", "violations", "CIS"]],
                   "row": picked.iloc[0].to_dict(),
                   "message": (f"{division}: top {len(picked)} by {metric_word} \u2014 worst is "
                               f"{picked.iloc[0]['name']} (CIS {picked.iloc[0]['CIS']:.1f}).")}
            if wants_route and len(picked) >= 2:
                rt = route_over(picked)
                if rt:
                    res["route_obj"] = rt
                    res["message"] += f" Shortest route \u2248 {rt['km']:.1f} km over {len(picked)} stops."
            return res

    area_df, area_label = match_area()
    if area_df is not None and not (has(LISTW) and n):
        area_df = area_df.sort_values(rank_col, ascending=False)
        if len(area_df) >= 2 and wants_route:
            picked = area_df.head(n or 6)
            res = {"intent": "division", "division": area_label,
                   "table": picked[["name", "violations", "CIS"]],
                   "row": picked.iloc[0].to_dict(),
                   "message": (f"{area_label}: {len(picked)} hotspots, worst "
                               f"{picked.iloc[0]['name']} (CIS {picked.iloc[0]['CIS']:.1f}).")}
            rt = route_over(picked)
            if rt:
                res["route_obj"] = rt
                res["message"] += f" Patrol route \u2248 {rt['km']:.1f} km."
            return res
        row = area_df.iloc[0]
        return {"intent": "focus_hotspot", "row": row.to_dict(),
                "message": (f"{row['name']} \u2014 CIS {row['CIS']:.1f}, "
                            f"{int(row['violations'])} violations, rank "
                            f"#{int(row['diff_rank'])} of {len(c)}.")}

    if has(IMPACT + VOLUME) and not has(LISTW) and n is None:
        top = c.sort_values(rank_col, ascending=False).iloc[0]
        return {"intent": "focus_hotspot", "row": top.to_dict(),
                "message": (f"Worst by {metric_word}: {top['name']} "
                            f"(CIS {top['CIS']:.1f}, {int(top['violations'])} violations).")}

    if n is not None or has(LISTW):
        kk = n or 10
        picked = c.sort_values(rank_col, ascending=False).head(kk)
        res = {"intent": "list",
               "table": picked[["diff_rank", "name", "violations", "CIS"]],
               "row": picked.iloc[0].to_dict(),
               "message": f"Top {kk} hotspots by {metric_word}."}
        if wants_route and len(picked) >= 2:
            rt = route_over(picked)
            if rt:
                res["route_obj"] = rt
                res["message"] += f" Route \u2248 {rt['km']:.1f} km."
        return res

    top5 = c.sort_values("CIS", ascending=False).head(5)
    return {"intent": "list", "table": top5[["diff_rank", "name", "violations", "CIS"]],
            "row": top5.iloc[0].to_dict(),
            "message": "I read that as a general query \u2014 here are the city\'s top 5 "
                       "congestion hotspots. Try a division (e.g. Cubbon Park), a junction, "
                       "\'worst\', \'top 5\', or \'morning route\'."}


# --------------------------------------------------------------------------- #
# Module 3 \u2014 challan-bias temporal view (honest version)
# --------------------------------------------------------------------------- #
def hourly_profile(temporal):
    if temporal is None:
        return pd.DataFrame({"hour": range(24), "raw": [0] * 24})
    h = temporal.groupby("hour")["count"].sum().reindex(range(24), fill_value=0)
    return pd.DataFrame({"hour": h.index, "raw": h.values})


def bias_view(temporal, active_hours=range(7, 22), gap_ratio=0.4):
    """Flag suspected ENFORCEMENT-GAP hours and provide an EXPLICITLY-LABELLED
    estimate of underlying demand. We do NOT claim this is ground truth — the
    timestamp is challan-creation time, so low hours during active daytime are
    most likely an enforcement gap, not absence of violations.

    Estimate = observed where plausible; for flagged gap hours we interpolate a
    floor from the active-hour median (a stated, conservative assumption)."""
    prof = hourly_profile(temporal)
    raw = prof["raw"].astype(float).values
    active = np.array([h in set(active_hours) for h in range(24)])
    active_median = np.median(raw[active]) if active.any() else raw.mean()
    floor = gap_ratio * active_median
    is_gap = active & (raw < floor)
    est = raw.copy()
    est[is_gap] = floor               # lift gap hours to the conservative floor
    prof["estimated_demand"] = est
    prof["is_enforcement_gap"] = is_gap
    gap_hours = [int(h) for h in prof["hour"][is_gap]]
    return prof, gap_hours


# --------------------------------------------------------------------------- #
# Module 4 — operational ROI
# --------------------------------------------------------------------------- #
def roi_metrics(routes, cis, schedule, total_violations=None):
    """Real, computed metrics (not mocked):
       - fleet route km
       - violation coverage: share of ALL violations at patrolled junctions
       - patrol efficiency: violations covered per route-km
    """
    out = {"route_km": 0.0, "cis_coverage_pct": 0.0,
           "violation_coverage_pct": 0.0, "patrolled_hotspots": 0,
           "violations_per_km": 0.0}
    if routes is not None and not routes.empty:
        per = routes.groupby([c for c in ["patrol", "shift"] if c in routes.columns]
                             or ["shift"]).agg(route_km=("cum_km", "max"))
        out["route_km"] = float(per["route_km"].sum())
    if schedule is not None and "hotspot" in schedule and cis is not None:
        patrolled = set(schedule["hotspot"].dropna().unique())
        sub = cis[cis["hotspot"].isin(patrolled)]
        denom = total_violations or cis["violations"].sum()
        cis_total = cis["CIS"].sum()
        out["cis_coverage_pct"] = round(100 * sub["CIS"].sum() / cis_total, 1) if cis_total else 0.0
        out["violation_coverage_pct"] = round(100 * sub["violations"].sum() / denom, 1) if denom else 0.0
        out["patrolled_hotspots"] = len(patrolled)
        if out["route_km"] > 0:
            out["violations_per_km"] = round(sub["violations"].sum() / out["route_km"], 0)
    return out


def roi_from_summary(route_summary):
    """If the pipeline's route summary (with naive_km) is available, use it for the
    exact, honest km-saved figure."""
    s = route_summary
    naive = float(s["naive_km"].sum())
    opt = float(s["route_km"].sum())
    saved = 100 * (naive - opt) / naive if naive else 0.0
    return {"naive_km": round(naive, 1), "route_km": round(opt, 1),
            "km_saved_pct": round(saved, 1)}


# --------------------------------------------------------------------------- #
# UI helpers (pydeck)
# --------------------------------------------------------------------------- #
def color_scale(values, lo=(40, 60, 120), hi=(220, 30, 30), alpha=190):
    """Map a numeric series to RGBA colours (cool -> hot)."""
    v = np.asarray(values, dtype=float)
    rng = v.max() - v.min()
    t = (v - v.min()) / rng if rng > 1e-9 else np.full_like(v, 0.5)
    cols = []
    for ti in t:
        cols.append([int(lo[0] + (hi[0]-lo[0])*ti),
                     int(lo[1] + (hi[1]-lo[1])*ti),
                     int(lo[2] + (hi[2]-lo[2])*ti), alpha])
    return cols


def map_frame(cis, metric):
    """Return a dataframe ready for a pydeck ColumnLayer for the chosen metric."""
    df = cis.copy()
    df = df.dropna(subset=["lat", "lon"])
    df["metric"] = df[metric].astype(float)
    df["elevation"] = 60 * df["metric"] / (df["metric"].max() or 1)  # 0..60 (x scale)
    df["color"] = color_scale(df["metric"])
    return df


def route_path(routes, shift):
    """Per-patrol [lon,lat] paths for a shift, for pydeck PathLayers."""
    if routes is None or routes.empty:
        return None
    sub = routes[routes["shift"] == shift]
    if sub.empty:
        return None
    paths = []
    if "patrol" in sub.columns:
        for p, g in sub.groupby("patrol"):
            g = g.sort_values("order")
            paths.append({"patrol": int(p),
                          "path": [[float(lo), float(la)]
                                   for la, lo in zip(g["lat"], g["lon"])]})
    else:
        g = sub.sort_values("order")
        paths.append({"patrol": 0,
                      "path": [[float(lo), float(la)]
                               for la, lo in zip(g["lat"], g["lon"])]})
    return {"shift": shift, "paths": paths,
            "stops": sub[["order", "name", "lat", "lon"]].to_dict("records")}
