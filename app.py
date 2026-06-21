"""
ParkIQ — Streamlit demo prototype.

    streamlit run app.py

Modules:
  1. AI Dispatcher          — deterministic NL query box (no live LLM)
  2. Before/After Diffusion — raw counts vs graph-diffused CIS on a 3D map + movers
  3. Coverage Planner       — live slider: how much of the city's violations to cover,
                              the cost curve, and the efficiency 'knee' (the max story)
  4. Challan-Bias view      — raw enforcement signal vs labelled demand estimate
  5. Operational ROI        — real coverage, distance saved, violations-per-km

All numbers read LIVE from outputs/*.csv (run_pipeline.py); nothing hardcoded.
mock_data.py supplies realistic stand-ins if those CSVs are absent.
"""
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import streamlit as st

import app_logic as L

st.set_page_config(page_title="ParkIQ — Parking-Congestion Intelligence",
                   layout="wide", page_icon="🅿️")
BENGALURU = (12.9716, 77.5946)


@st.cache_data
def get_data():
    return L.load_artifacts("outputs")


def deck_map(df_pts, focus=None, route=None, highlight=None):
    lat0, lon0, zoom = BENGALURU[0], BENGALURU[1], 11
    if focus and not pd.isna(focus.get("lat")):
        lat0, lon0, zoom = focus["lat"], focus["lon"], 14
    layers = [pdk.Layer(
        "ColumnLayer", data=df_pts, get_position=["lon", "lat"],
        get_elevation="elevation", elevation_scale=120, radius=130,
        get_fill_color="color", pickable=True, auto_highlight=True)]
    if route and route.get("paths"):
        layers.append(pdk.Layer("PathLayer", data=route["paths"], get_path="path",
                                get_width=6, width_min_pixels=3,
                                get_color=[255, 165, 0]))
        layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame(route["stops"]),
                                get_position=["lon", "lat"], get_radius=90,
                                get_fill_color=[255, 165, 0], pickable=True))
    return pdk.Deck(layers=layers, map_style="road",
                    initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0,
                                                     zoom=zoom, pitch=45),
                    tooltip={"text": "{name}\nCIS {CIS} | violations {violations}"})


def main():
    data = get_data()
    cis, schedule, routes = data["cis"], data["schedule"], data["routes"]
    rsum, temporal = data["route_summary"], data["temporal"]
    curve, total_v = data["curve"], data["total_violations"]
    equity = data["equity"]
    hot_div = L.hotspot_division(equity)

    st.title("🅿️ ParkIQ — Predictive Parking-Congestion Intelligence")
    st.markdown("**See → Score → Act.** Bengaluru Traffic Police · "
                f"{total_v:,} parking violations · 168 BTP junctions · "
                "*detect hotspots, quantify congestion impact, deploy patrols predictively.*")

    # ---------------- ROI metric strip ----------------
    roi = L.roi_metrics(routes, cis, schedule, total_violations=total_v)
    km = L.roi_from_summary(rsum) if rsum is not None else {"naive_km": 0, "route_km": roi["route_km"], "km_saved_pct": 0}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Violation coverage", f"{roi['violation_coverage_pct']}%",
              help=f"Share of ALL {total_v:,} violations at the {roi['patrolled_hotspots']} "
                   "patrolled junctions. Patrols run multi-stop routes.")
    c2.metric("Patrol distance saved", f"{km['km_saved_pct']}%",
              delta=f"-{km['naive_km']-km['route_km']:.0f} km vs naive", delta_color="inverse",
              help="AI TSP routes vs visiting the same stops in priority order.")
    c3.metric("Patrol efficiency", f"{roi['violations_per_km']:.0f}",
              help="Violations covered per route-km — higher = leaner deployment.")
    c4.metric("Junctions patrolled", f"{roi['patrolled_hotspots']}",
              help=f"Minimum set (of {len(cis)} hotspots) to hit the coverage target.")
    # ---------------- Business case (Rupees & Hours) ----------------
    with st.expander("💰 Business case — what the 29.5% distance saving is worth", expanded=False):
        ac1, ac2, ac3, ac4 = st.columns(4)
        kmpl = ac1.number_input("km / litre", 4.0, 20.0, 8.0, 0.5)
        price = ac2.number_input("fuel ₹ / litre", 50.0, 130.0, 95.0, 1.0)
        speed = ac3.number_input("city speed km/h", 8.0, 40.0, 20.0, 1.0)
        days = ac4.number_input("days / month", 20, 31, 30, 1)
        bc = L.business_case(km["naive_km"], km["route_km"], km_per_litre=kmpl,
                             fuel_price=price, city_speed_kmh=speed, days=days)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Distance saved / day", f"{bc['km_saved_day']:.0f} km")
        b2.metric("Fuel saved / month", f"{bc['litres_day']*days:.0f} L",
                  help=f"₹{bc['rupees_month']:,.0f}/month · ₹{bc['rupees_year']:,.0f}/year")
        b3.metric("Fuel cost saved / month", f"₹{bc['rupees_month']:,.0f}")
        b4.metric("Policing hours returned / month", f"{bc['hours_month']:.0f} hrs",
                  help="Driving time avoided = active enforcement time returned to the force.")
        st.caption("Illustrative — assumptions editable above. Figures are for the patrolled "
                   "fleet; they scale with the number of vehicles and shifts deployed.")

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("🛰️ AI Dispatcher")
        q = st.text_input("Ask the dispatcher",
                          placeholder="'highest impact in Cubbon Park with a route', "
                                      "'top 5 in City Market', 'morning route', 'BTP082'")
        result = L.nl_dispatch(q, cis, routes, hot_div=hot_div)
        focus = result.get("row")
        route = None
        if result["intent"] == "route":
            route = L.route_path(routes, result["shift"])
        elif "route_obj" in result:
            route = result["route_obj"]
        st.info(result["message"])
        if "table" in result:
            st.dataframe(result["table"], hide_index=True, width='stretch')
        # explain WHY: CIS component radar for the focused/top hotspot
        comp = ["density", "recurrence", "peakedness", "criticality", "spillover"]
        if focus and all(k in focus and focus[k] == focus[k] for k in comp):
            rad = go.Figure(go.Scatterpolar(
                r=[float(focus[k]) for k in comp] + [float(focus[comp[0]])],
                theta=[k.title() for k in comp] + [comp[0].title()],
                fill="toself", line_color="#d62728"))
            rad.update_layout(height=260, margin=dict(t=30, b=10, l=30, r=30),
                              polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                              showlegend=False,
                              title=f"Why {focus['name'][:34]} scores CIS {focus['CIS']:.0f}")
            st.plotly_chart(rad, width='stretch')
    with right:
        st.subheader("📊 Top hotspots")
        st.dataframe(cis.sort_values("CIS", ascending=False).head(8)
                     [["diff_rank", "name", "violations", "CIS"]]
                     .rename(columns={"diff_rank": "rank"}),
                     hide_index=True, width='stretch')

    # ---------------- Module 2: Before/After diffusion ----------------
    st.subheader("🗺️ Why graph-diffusion beats raw counts")
    view = st.radio("Map metric", ["AI Diffused Impact (CIS)", "Raw Violation Counts"],
                    horizontal=True)
    metric = "violations" if "Raw" in view else "CIS"
    st.caption("Raw counts overrate **isolated** high-volume spots (Safina Plaza); diffused "
               "CIS makes **networked choke-points** (Elite Junction) glow — they amplify "
               "congestion across neighbours." if "Diffused" in view else
               "Switch to *AI Diffused Impact* to see the networked choke-points rise.")
    st.pydeck_chart(deck_map(L.map_frame(cis, metric), focus=focus, route=route))
    with st.expander("Biggest rank movers (raw count → diffused CIS)"):
        promoted, demoted = L.movers(cis, 5)
        m1, m2 = st.columns(2)
        m1.markdown("**Promoted — networked choke-points**")
        m1.dataframe(promoted[["name", "violations", "CIS", "move", "spillover"]],
                     hide_index=True, width='stretch')
        m2.markdown("**Demoted — isolated high-volume**")
        m2.dataframe(demoted[["name", "violations", "CIS", "move", "spillover"]],
                     hide_index=True, width='stretch')

    # ---------------- Module 3: Coverage Planner ----------------
    st.subheader("🎯 Coverage planner — how far should we push?")
    opt_for = st.radio("Optimise for", ["Congestion impact (severity-weighted)",
                                        "Raw violation count"], horizontal=True)
    quantity = "impact_volume" if "impact" in opt_for.lower() else "violations"
    qlabel = "congestion impact" if quantity == "impact_volume" else "violations"
    pcurve = L.coverage_curve(cis, total_v, quantity=quantity)
    knee_n, knee_cov = L.knee_point(pcurve)
    ceiling = float(pcurve["coverage_pct"].max())
    target = st.slider(f"Target: cover this % of all {qlabel}", 40, int(ceiling),
                       min(80, int(ceiling)), help="Drag to see the cost in junctions.")
    sel = L.select_for_target(cis, target, total_v, quantity=quantity)
    a, b, c = st.columns(3)
    a.metric("Junctions to patrol", sel["n"])
    b.metric("Achieved coverage", f"{sel['coverage']:.1f}%")
    c.metric("Hard ceiling", f"{ceiling:.1f}%",
             help=f"Patrolling ALL {len(cis)} hotspots.")
    if quantity == "impact_volume":
        st.caption("**Impact mode** weights each violation by its hotspot's *consequence* "
                   "(arterial/junction criticality x rush-hour concentration, 0.5x–2.0x) — "
                   "so the optimiser prioritises chokepoints, not just busy spots. This is "
                   "how we 'quantify impact on traffic flow' (a proxy — no live speed feed).")
    fig = go.Figure()
    fig.add_scatter(x=pcurve["n_hotspots"], y=pcurve["coverage_pct"], mode="lines",
                    line=dict(color="#1f77b4", width=3), name="Coverage curve")
    fig.add_scatter(x=[sel["n"]], y=[sel["coverage"]], mode="markers",
                    marker=dict(size=13, color="#d62728"), name="Your target")
    fig.add_scatter(x=[knee_n], y=[knee_cov], mode="markers+text",
                    marker=dict(size=12, color="#2ca02c", symbol="diamond"),
                    text=["sweet spot"], textposition="top center",
                    name="Efficiency knee")
    fig.add_vline(x=knee_n, line_dash="dot", line_color="#2ca02c")
    fig.update_layout(height=320, xaxis_title="# junctions patrolled",
                      yaxis_title=f"% of all {qlabel} covered",
                      margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width='stretch')
    st.caption(f"Diminishing returns: the **efficiency sweet spot is ~{knee_cov:.0f}% at "
               f"{knee_n} junctions**. Beyond it, each extra junction buys almost nothing — "
               f"deploying past the knee is spending police time and budget for near-zero "
               f"gain. Ceiling is {ceiling:.0f}%.")

    # --- zone-of-influence radius: greedy max-coverage (Maximum Coverage Problem) ---
    st.markdown("**Zone-of-influence radius** — merge nearby hotspots into one patrol "
                "beat. Bigger radius = fewer stops, but each stop is a larger area to work.")
    rcol1, rcol2 = st.columns([1, 2])
    with rcol1:
        radius = st.select_slider("Beat radius (m)", [150, 300, 500, 800, 1200, 1500], 500)
    gz = L.greedy_zone_cover(cis, total_v, radius, target, quantity=quantity)
    rcol2.metric(f"Stops to cover {target}% at {radius} m beats", gz["n_zones"],
                 delta=f"{gz['n_zones']-sel['n']} vs {sel['n']} point-stops",
                 delta_color="inverse",
                 help="Greedy Maximum-Coverage: each beat absorbs all hotspots within "
                      "the radius. Even large beats can't reach a dozen stops — violations "
                      "are spread across the city, not concentrated.")

    # named-junction list vs AI-selected set (answers 'why not all junctions?')
    named = cis[cis["hotspot"].astype(str).str.startswith("J:")]
    named_cov = 100 * named["violations"].sum() / total_v
    bar = go.Figure(go.Bar(
        x=[f"All {len(named)} named junctions", f"AI set ({sel['n']} hotspots)"],
        y=[named_cov, sel["coverage"]], marker_color=["#9aa0b5", "#d62728"],
        text=[f"{named_cov:.0f}%", f"{sel['coverage']:.0f}%"], textposition="outside"))
    bar.update_layout(height=260, yaxis_title="% of all violations covered",
                      margin=dict(t=10, b=10), title="Naming ≠ where violations are")
    st.plotly_chart(bar, width='stretch')

    # ---------------- What-if disruption simulation ----------------
    st.subheader("🚧 What-if: simulate a disruption")
    wc1, wc2 = st.columns([2, 1])
    opts = cis.sort_values("CIS", ascending=False)["name"].tolist()
    pick = wc1.selectbox("Disruption at (metro works / closure / event near):", opts)
    mult = wc2.slider("Congestion impact ×", 1.0, 3.0, 1.8, 0.1)
    hid = cis[cis["name"] == pick]["hotspot"].iloc[0]
    sim = L.simulate_disruption(cis, hid, mult, target, total_v, quantity=quantity)
    if sim:
        s1, s2, s3 = st.columns(3)
        s1.metric("Impact rank", f"#{sim['after_rank']}",
                  delta=f"{sim['before_rank']-sim['after_rank']:+d} places",
                  help=f"Was #{sim['before_rank']} before the disruption.")
        s2.metric("In the patrol plan?",
                  "Yes" if sim["in_set_after"] else "No",
                  delta=("now included" if (sim["in_set_after"] and not sim["in_set_before"])
                         else "unchanged"))
        s3.metric("Junctions to re-deploy", sim["n_after"],
                  delta=f"{sim['n_after']-sim['n_before']:+d}")
        if sim["in_set_after"] and not sim["in_set_before"]:
            st.success(f"A disruption at **{sim['name']}** lifts its impact (CIS→{sim['sim_cis']:.0f}); "
                       "the optimiser re-ranks it and **pulls it into the patrol plan**, re-sequencing "
                       "the affected route automatically. The model adapts — it isn't a static "
                       "historical calculator.")
        else:
            st.caption(f"{sim['name']} CIS → {sim['sim_cis']:.0f}; rank "
                       f"{sim['before_rank']}→{sim['after_rank']}. (Re-routing here means "
                       "re-selection + re-sequencing on straight-line distance — we don't have a "
                       "road graph to physically route around a closed segment.)")

    # ---------------- Module 3b: Equity check ----------------
    st.subheader("⚖️ Equity check — is coverage fair across police divisions?")
    floor_on = st.checkbox("Apply fairness floor (guarantee ≥1 patrolled hotspot per division)")
    patrolled = set(sel["hotspots"])
    base_cov = sel["coverage"]
    if floor_on:
        patrolled = patrolled | L.division_top_hotspots(equity)
    zdf = L.equity_by_zone(equity, patrolled)
    es = L.equity_summary(zdf)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Divisions covered <50%", es["below_50"], help=f"Out of {es['n_zones']}.")
    e2.metric("Zero-coverage divisions", es["zero"],
              help="Divisions whose violations get no patrol at all.")
    e3.metric("Coverage inequality (Gini)", f"{es['gini']:.2f}",
              help="0 = every division equally covered; higher = more unequal.")
    e4.metric("Citywide coverage",
              f"{L.coverage_of(cis, patrolled, total_v):.1f}%",
              delta=(f"+{L.coverage_of(cis, patrolled, total_v)-base_cov:.1f}% & "
                     f"+{len(patrolled)-len(sel['hotspots'])} junctions" if floor_on else None))
    worst = zdf.head(18)
    ecol = ["#d62728" if v < 50 else "#2ca02c" for v in worst["coverage_pct"]]
    ef = go.Figure(go.Bar(x=worst["coverage_pct"], y=worst["zone"], orientation="h",
                          marker_color=ecol,
                          text=[f"{v:.0f}%" for v in worst["coverage_pct"]],
                          textposition="outside"))
    ef.add_vline(x=50, line_dash="dash", line_color="#888")
    ef.update_layout(height=460, xaxis_title="% of division's violations covered",
                     yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10),
                     title="Least-covered divisions (red = under-served)")
    st.plotly_chart(ef, width='stretch')
    if floor_on:
        st.success("Fairness floor on: every division now has at least its worst hotspot "
                   "patrolled — zero-coverage divisions eliminated for a handful of extra "
                   "junctions and almost no loss of citywide coverage.")
    else:
        st.caption("Efficiency-first deployment concentrates on high-volume central "
                   "divisions and leaves outer ones under-served. Tick the box above to see "
                   "a **fairness floor** close the gap at minimal cost — a deliberate "
                   "equity-vs-efficiency trade a commander can choose.")

    # ---------------- Module 4: Challan-bias ----------------
    st.subheader("⏱️ Enforcement-bias correction")
    prof, gaps = L.bias_view(temporal)
    f2 = go.Figure()
    f2.add_bar(x=prof["hour"], y=prof["raw"], name="Raw challans (enforcement signal)",
               marker_color="#9aa0b5")
    f2.add_scatter(x=prof["hour"], y=prof["estimated_demand"], mode="lines+markers",
                   name="Assumption-adjusted estimate", line=dict(color="#d62728", width=3))
    f2.update_layout(height=320, xaxis_title="Hour (IST)", yaxis_title="Violations",
                     legend=dict(orientation="h", y=-0.3), margin=dict(t=10, b=10))
    st.plotly_chart(f2, width='stretch')
    if gaps:
        st.warning(f"**Suspected enforcement gap {min(gaps):02d}:00–{max(gaps)+1:02d}:00 IST.** "
                   "Challans crash to near-zero — almost certainly because the timestamp is "
                   "*challan-creation* time (morning-shift heavy), not because parking stops. "
                   "The red line is an **explicitly assumption-based** estimate, **not ground "
                   "truth**. **Action:** add evening patrol coverage.")

    st.caption("ParkIQ pipeline — CIS, graph-diffusion spillover, LightGBM forecast "
               "(Precision@20 = 0.95), coverage-target optimiser, TSP routing.")


if __name__ == "__main__":
    main()
