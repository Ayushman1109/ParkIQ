"""
features.py — temporal features + spatial hotspot mining.

Hotspot definition:
  * If lat/lon exist  -> HDBSCAN on (lat,lon) in radians w/ haversine metric.
    Each dense cluster = one hotspot. Noise points (-1) are dropped from the
    hotspot layer (they're sporadic, not chronic) but kept for city-wide stats.
  * Else               -> the location (or zone) string IS the spatial unit.
"""
import ast
import numpy as np
import pandas as pd


def add_temporal(df: pd.DataFrame, rush_hours) -> pd.DataFrame:
    df = df.copy()
    ts = df["ts"]
    df["date"] = ts.dt.date
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = ts.dt.month
    df["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    df["is_rush"] = df["hour"].isin(rush_hours).astype(int)
    return df


def apply_peak_hours(df, mode="data_driven", fixed_rush=None):
    """Set df['is_rush'] either from fixed commute windows or from the data's own
    busy hours (hours with above-average volume). Returns the peak-hour set.

    Why data-driven: in this dataset the timestamp is challan-creation time
    (morning-shift heavy), so assuming 5-8pm commute rush would mislabel the real
    peak. Letting the data speak is more robust to timestamp semantics."""
    if mode == "data_driven":
        by_hour = df["hour"].value_counts()
        peak = set(by_hour[by_hour > by_hour.mean()].index.tolist())
    else:
        peak = set(fixed_rush or [])
    df = df.copy()
    df["is_rush"] = df["hour"].isin(peak).astype(int)
    return df, sorted(peak)


def _parse_offences(val):
    """violation_type may be a JSON-style list string, or a plain label."""
    if isinstance(val, list):
        return [str(x).strip().strip('"') for x in val]
    if not isinstance(val, str):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip().strip('"') for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [val.strip()]


def add_offence_severity(df, schema, severity_map, default):
    """Per-record worst-offence severity (drives CIS criticality)."""
    df = df.copy()
    col = schema.get("violation_type")
    if not col:
        df["_severity"] = default
        return df
    sev = []
    for v in df[col].values:
        offs = _parse_offences(v)
        if not offs:
            sev.append(default)
        else:
            sev.append(max(severity_map.get(o.upper(), default) for o in offs))
    df["_severity"] = sev
    return df


def _grid_cells(lat, lon, step_m):
    mlat = 111_320.0
    mlon = 111_320.0 * np.cos(np.radians(np.nanmedian(lat)))
    slat, slon = step_m / mlat, step_m / mlon
    clat = np.floor(lat / slat).astype("int64")
    clon = np.floor(lon / slon).astype("int64")
    cell = pd.Series([f"{a}_{b}" for a, b in zip(clat, clon)], index=lat.index)
    return cell, clat.values, clon.values


def _merge_adjacent(big_cells, connectivity=4):
    """Union-find: merge adjacent hot cells into one hotspot.
    connectivity=4 (orthogonal only) reduces diagonal chaining into mega-blobs."""
    parent = {c: c for c in big_cells}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if connectivity == 8:
        nbrs = [(da, db) for da in (-1, 0, 1) for db in (-1, 0, 1)
                if not (da == 0 and db == 0)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for (a, b) in big_cells:
        for da, db in nbrs:
            nb = (a + da, b + db)
            if nb in parent:
                union((a, b), nb)
    roots, comp = {}, {}
    for c in big_cells:
        r = find(c)
        if r not in roots:
            roots[r] = f"G{len(roots):04d}"
        comp[c] = roots[r]
    return comp


def _merged_grid_ids(cells, clat, clon, min_cluster_size):
    """Map each row -> merged grid hotspot id, or '-1' if in a sparse cell."""
    sizes = cells.value_counts()
    big = {tuple(int(p) for p in s.split("_"))
           for s in sizes[sizes >= min_cluster_size].index}
    comp = _merge_adjacent(big, connectivity=4)
    return np.array([comp.get((cl, co), "-1") for cl, co in zip(clat, clon)])


def cluster_hotspots(df, schema, method="auto", grid_m=150,
                     min_cluster_size=60, radius_m=120, hdbscan_max=60000):
    """Return (df+hotspot col, hotspots meta DataFrame, mode string)."""
    has_geo = bool(schema.get("latitude") and schema.get("longitude"))
    df = df.copy()

    if method == "auto":
        method = ("junction" if has_geo and schema.get("junction")
                  else "grid" if has_geo and len(df) > hdbscan_max
                  else "hdbscan" if has_geo else "names")

    if has_geo and method in ("grid", "hdbscan", "junction"):
        lat = pd.to_numeric(df[schema["latitude"]], errors="coerce")
        lon = pd.to_numeric(df[schema["longitude"]], errors="coerce")
        ok = lat.notna() & lon.notna()
        df = df[ok].copy()
        df["_lat"] = lat[ok].values
        df["_lon"] = lon[ok].values

        if method == "junction":
            # Hotspot = the official BTP junction for rows that have one;
            # the rest are grid-celled (merged) so we still catch off-junction spots.
            jcol = schema["junction"]
            jname = df[jcol].astype(str)
            at_jct = ~jname.str.contains("No Junction", case=False, na=False)
            cells, clat, clon = _grid_cells(df["_lat"], df["_lon"], grid_m)
            grid_id = _merged_grid_ids(cells, clat, clon, min_cluster_size)
            hotspot = np.where(at_jct, "J:" + jname, grid_id)
            df["hotspot"] = hotspot
            # enforce min size on junction hotspots too
            counts = pd.Series(hotspot).value_counts()
            small = set(counts[counts < min_cluster_size].index)
            df["hotspot"] = df["hotspot"].where(~df["hotspot"].isin(small), "-1")
            mode = "geo"
        elif method == "grid":
            cells, clat, clon = _grid_cells(df["_lat"], df["_lon"], grid_m)
            df["hotspot"] = _merged_grid_ids(cells, clat, clon, min_cluster_size)
            mode = "geo"
        else:  # hdbscan
            import hdbscan
            coords = np.radians(np.c_[df["_lat"].values, df["_lon"].values])
            cl = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                                 metric="haversine",
                                 cluster_selection_epsilon=radius_m / 6_371_000.0)
            df["hotspot"] = cl.fit_predict(coords).astype(str)
            mode = "geo"

        hot = df[df["hotspot"].astype(str) != "-1"]
        meta = (hot.groupby("hotspot")
                .agg(lat=("_lat", "mean"), lon=("_lon", "mean"),
                     n=("hotspot", "size")).reset_index())
        meta["name"] = _label_hotspots(hot, schema, meta["hotspot"])
        meta["has_junction"] = _junction_flag(hot, schema, meta["hotspot"])
    else:
        unit_col = schema.get("location") or schema.get("zone")
        if not unit_col:
            raise ValueError("No lat/lon AND no location/zone column. Set one in "
                             "config.COLUMN_MAP.")
        df["hotspot"] = df[unit_col].astype(str)
        meta = df.groupby("hotspot").size().rename("n").reset_index()
        meta["name"] = meta["hotspot"]
        meta["lat"] = np.nan
        meta["lon"] = np.nan
        meta["has_junction"] = 0.0
        mode = "names"

    return df, meta, mode


def _label_hotspots(hot, schema, hotspot_ids):
    """Human label per hotspot = dominant real junction, else modal location."""
    jcol = schema.get("junction")
    lcol = schema.get("location")
    labels = {}
    for hid, sub in hot.groupby("hotspot"):
        label = None
        if jcol:
            j = sub[jcol].astype(str)
            real = j[~j.str.contains("No Junction", case=False, na=False)]
            if len(real):
                label = real.mode().iloc[0]
        if label is None and lcol:
            loc = sub[lcol].astype(str)
            label = loc.mode().iloc[0][:48] if len(loc) else hid
        labels[hid] = label or f"Hotspot {hid}"
    return hotspot_ids.map(labels)


def _junction_flag(hot, schema, hotspot_ids):
    """Share of a hotspot's violations sitting at a named junction (0-1)."""
    jcol = schema.get("junction")
    if not jcol:
        return pd.Series(0.0, index=hotspot_ids.index)
    share = {}
    for hid, sub in hot.groupby("hotspot"):
        j = sub[jcol].astype(str)
        share[hid] = float((~j.str.contains("No Junction", case=False, na=False)).mean())
    return hotspot_ids.map(share)
