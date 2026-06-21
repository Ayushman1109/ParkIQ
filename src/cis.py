"""
cis.py — Congestion Impact Score (the headline innovation).

The problem statement asks us to *quantify the impact on traffic flow* so
enforcement can be *prioritised*. A raw violation count can't do that: a spot with
500 scattered 3am violations is not a congestion problem; 200 violations packed into
weekday rush hour at a metro junction is. CIS encodes that judgement transparently.

For each hotspot we compute five 0–1 components, then a weighted blend → 0–100:

  Density      how much offending volume (log-scaled so megaspots don't dominate)
  Recurrence   how chronic — share of days the spot was active (habitual vs one-off)
  Peakedness   share of offending inside rush windows (peak-hour blocking = worst)
  Criticality  how congestion-critical the road is (POI-keyword proxy; OSM-ready)
  Spillover    how much it co-offends with neighbours (clustered blocking spreads)

Every score ships with its component breakdown, so enforcement officers see *why*
a spot ranks high — not a black box. That explainability is a deliberate design
choice for a public-sector decision tool.
"""
import numpy as np
import pandas as pd


def _minmax(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else pd.Series(0.5, index=x.index)


def _criticality(name: str, keywords) -> float:
    if not isinstance(name, str):
        return 0.0
    n = name.lower()
    hits = sum(1 for kw in keywords if kw in n)
    return min(hits / 2.0, 1.0)   # 2+ keyword hits saturates


def _graph_diffusion_spillover(out, tau_km=0.5, cutoff_km=1.5, hops=2, decay=0.5):
    """Kernel-weighted neighbourhood diffusion of violation intensity.

    Edge weight w_ij = exp(-(d_ij/tau)^2) for d_ij < cutoff (Gaussian spatial
    kernel). Intensity v = log1p(violations). Spillover received at node i:
        s = (W + decay * W^2) @ v_norm     (1- and 2-hop contributions)
    Pure linear algebra on the hotspot graph — scales fine to thousands of nodes.
    """
    from sklearn.metrics.pairwise import haversine_distances
    coords = np.radians(out[["lat", "lon"]].fillna(0).values)
    d = haversine_distances(coords) * 6371.0          # km
    W = np.exp(-(d / tau_km) ** 2)
    W[d >= cutoff_km] = 0.0
    np.fill_diagonal(W, 0.0)
    v = np.log1p(out["violations"].values.astype(float))
    v = v / (v.max() + 1e-9)
    s = W @ v
    if hops >= 2:
        s = s + decay * (W @ (W @ v))
    return s


def compute_cis(df, meta, mode, weights, rush_hours, critical_keywords):
    """df: violations with 'hotspot','date','is_rush'. Returns ranked CIS table."""
    work = df[df["hotspot"].astype(str) != "-1"].copy()
    g = work.groupby("hotspot")

    n_days_total = work["date"].nunique()

    density_raw = g.size().rename("violations")
    recurrence_raw = g["date"].nunique().rename("active_days") / max(n_days_total, 1)
    peak_raw = g["is_rush"].mean().rename("rush_share")

    out = pd.concat([density_raw, recurrence_raw, peak_raw], axis=1).reset_index()
    keep_meta = [c for c in ["hotspot", "name", "lat", "lon", "has_junction"]
                 if c in meta.columns]
    out = out.merge(meta[keep_meta], on="hotspot", how="left")

    # --- criticality: real junction presence + offence severity -------------
    # (falls back to a location-keyword proxy only if neither signal exists)
    if "_severity" in work.columns:
        sev = work.groupby("hotspot")["_severity"].mean()
        out["severity_mean"] = out["hotspot"].map(sev).fillna(0.4)
    else:
        out["severity_mean"] = 0.4
    if "has_junction" in out.columns and out["has_junction"].notna().any():
        out["has_junction"] = out["has_junction"].fillna(0.0)
        out["criticality_raw"] = 0.5 * out["has_junction"] + 0.5 * out["severity_mean"]
    else:
        out["criticality_raw"] = out["name"].apply(
            lambda s: _criticality(s, critical_keywords))

    # --- spillover: graph diffusion over the hotspot network ----------------
    # We model hotspots as nodes and proximity as weighted edges, then let
    # offending "intensity" diffuse to neighbours (2 hops). A node surrounded by
    # heavy nearby offending carries high spillover risk — an unsupervised,
    # network-structural measure of cascading impact (no traffic-flow labels
    # invented; see README on why a supervised STGCN isn't honest on this data).
    if mode == "geo" and out["lat"].notna().any():
        out["spillover_raw"] = _graph_diffusion_spillover(out)
    else:
        vt = work.groupby("hotspot").size()
        out["spillover_raw"] = out["hotspot"].map(vt).fillna(0)

    # --- normalise components ----------------------------------------------
    out["density"] = _minmax(np.log1p(out["violations"]))
    out["recurrence"] = out["active_days"].clip(0, 1)
    out["peakedness"] = out["rush_share"].clip(0, 1)
    out["criticality"] = out["criticality_raw"].clip(0, 1)
    out["spillover"] = _minmax(out["spillover_raw"])

    # --- blend --------------------------------------------------------------
    out["CIS"] = 100 * sum(weights[k] * out[k] for k in weights)
    out = out.sort_values("CIS", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    cols = ["rank", "hotspot", "name", "lat", "lon", "violations",
            "CIS", "density", "recurrence", "peakedness", "criticality", "spillover"]
    return out[cols].round(3)
