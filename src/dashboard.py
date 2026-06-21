"""
dashboard.py — the "SEE" layer. Builds a self-contained interactive HTML report:
  * Folium map: violation heatmap + CIS-ranked hotspot markers (geo mode)
  * Plotly: hour-of-day x day-of-week offending matrix, monthly trend,
    and a CIS component breakdown for the worst spots.
Everything writes to outputs/ as standalone files you can open in any browser
(or screenshot straight into the pitch deck).
"""
import os
import numpy as np
import pandas as pd


def _color(cis):
    if cis >= 75:   return "darkred"
    if cis >= 50:   return "red"
    if cis >= 30:   return "orange"
    return "green"


def build_map(df, cis_table, mode, out_path, routes_df=None):
    if mode != "geo" or cis_table["lat"].isna().all():
        print("[dashboard] No geo coords — skipping map (charts still generated).")
        return None
    import folium
    from folium.plugins import HeatMap

    work = df[df["hotspot"].astype(str) != "-1"]
    center = [work["_lat"].mean(), work["_lon"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    HeatMap(work[["_lat", "_lon"]].dropna().values.tolist(),
            radius=12, blur=18, name="Violation density").add_to(m)

    for _, r in cis_table.dropna(subset=["lat", "lon"]).iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=6 + 10 * (r["CIS"] / 100),
            color=_color(r["CIS"]), fill=True, fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>#{int(r['rank'])} {r['name']}</b><br>"
                f"CIS: <b>{r['CIS']:.1f}</b><br>"
                f"Violations: {int(r['violations'])}<br>"
                f"density {r['density']:.2f} | recurrence {r['recurrence']:.2f}<br>"
                f"peak {r['peakedness']:.2f} | critical {r['criticality']:.2f} | "
                f"spill {r['spillover']:.2f}", max_width=260),
        ).add_to(m)

    # patrol routes, each its own toggleable coloured layer
    if routes_df is not None and not routes_df.empty:
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728",
                  "#17becf", "#bcbd22", "#e377c2", "#8c564b", "#7f7f7f"]
        group_key = "patrol" if "patrol" in routes_df.columns else "shift"
        for i, (gid, grp) in enumerate(routes_df.groupby(group_key)):
            col = colors[i % len(colors)]
            fg = folium.FeatureGroup(name=f"{group_key.title()} {gid}", show=False)
            # draw each (patrol,shift) leg sequence as a polyline
            sub_keys = ["shift"] if group_key == "patrol" else ["patrol"] \
                if "patrol" in grp.columns else []
            if sub_keys:
                for _, sg in grp.groupby(sub_keys):
                    sg = sg.sort_values("order")
                    folium.PolyLine(sg[["lat", "lon"]].values.tolist(),
                                    color=col, weight=3, opacity=0.8).add_to(fg)
            else:
                g = grp.sort_values("order")
                folium.PolyLine(g[["lat", "lon"]].values.tolist(),
                                color=col, weight=3, opacity=0.8).add_to(fg)
            for _, r in grp.iterrows():
                folium.CircleMarker([r["lat"], r["lon"]], radius=4, color=col,
                                    fill=True, fill_opacity=0.9,
                                    popup=f"{group_key} {gid}: {r['name']}").add_to(fg)
            fg.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_path)
    print(f"[dashboard] map -> {out_path}")
    return out_path


def build_charts(df, cis_table, out_path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    work = df[df["hotspot"].astype(str) != "-1"].copy()

    # hour x dow matrix
    mat = (work.groupby(["dow", "hour"]).size()
           .reset_index(name="n")
           .pivot(index="dow", columns="hour", values="n").fillna(0))
    dows = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Offending heatmap: hour x weekday",
                        "Monthly trend",
                        "Top-10 hotspots by CIS",
                        "CIS breakdown (worst 5 spots)"),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.16, horizontal_spacing=0.12)

    fig.add_trace(go.Heatmap(z=mat.values, x=[f"{h:02d}" for h in mat.columns],
                             y=[dows[i] for i in mat.index], colorscale="YlOrRd",
                             showscale=False), 1, 1)

    monthly = work.groupby(work["ts"].dt.to_period("M").astype(str)).size()
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values, mode="lines+markers",
                             line=dict(width=3)), 1, 2)

    top = cis_table.head(10)
    fig.add_trace(go.Bar(x=top["CIS"], y=top["name"], orientation="h",
                         marker_color="indianred"), 2, 1)

    worst = cis_table.head(5)
    comps = ["density", "recurrence", "peakedness", "criticality", "spillover"]
    for c in comps:
        fig.add_trace(go.Bar(name=c, x=worst["name"], y=worst[c]), 2, 2)
    fig.update_layout(barmode="stack", height=820, showlegend=True,
                      title_text="ParkIQ — Parking-Congestion Intelligence",
                      legend=dict(orientation="h", y=-0.08))
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[dashboard] charts -> {out_path}")
    return out_path
