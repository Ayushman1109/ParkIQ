"""
loader.py — robust loading + automatic schema detection + parking filter.

Why auto-detection: we don't know the exact column names in the real HackerEarth
file. This guesses them from name patterns + value checks, prints what it found,
and lets config.COLUMN_MAP override anything it gets wrong.
"""
import re
import sys
import warnings
import numpy as np
import pandas as pd

DATE_PAT = re.compile(r"date|day|dt$", re.I)
TIME_PAT = re.compile(r"time|hour|hrs", re.I)
DTYPE_PAT = re.compile(r"timestamp|datetime|occurr|reported|event", re.I)
LAT_PAT = re.compile(r"lat", re.I)
LON_PAT = re.compile(r"lon|lng", re.I)
VIO_PAT = re.compile(r"violat|offen[cs]e|act|charge|breach|nature", re.I)
ID_PAT = re.compile(r"(^|_)id($|_)|number|\bno\b|uid|fir|challan", re.I)
LOC_PAT = re.compile(r"locat|place|road|junction|spot|address|landmark|street", re.I)
ZONE_PAT = re.compile(r"zone|ward|division|circle|station|district|region|area", re.I)
VEH_PAT = re.compile(r"vehicle|class|veh_?type", re.I)
FINE_PAT = re.compile(r"fine|amount|penalty|fee", re.I)


def load_raw(path: str) -> pd.DataFrame:
    """Read csv / xlsb / xlsx / parquet defensively (engine by extension)."""
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("xlsb", "xlsx", "xls", "xlsm"):
        engine = "pyxlsb" if ext == "xlsb" else None
        df = pd.read_excel(path, engine=engine, sheet_name=0)
    elif ext == "parquet":
        df = pd.read_parquet(path)
    else:
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False,
                                 on_bad_lines="skip")
                break
            except UnicodeDecodeError:
                df = None
        if df is None:
            raise IOError(f"Could not read {path} with utf-8 or latin-1.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _first_match(cols, pat, df=None, numeric=False, latlon=None, prefer_text=False):
    for c in cols:
        if not pat.search(c):
            continue
        if prefer_text:
            # skip identifier columns and near-unique columns (those are IDs)
            if ID_PAT.search(c):
                continue
            if df is not None and df[c].nunique(dropna=True) > 0.5 * len(df):
                continue
        if numeric and df is not None and not pd.api.types.is_numeric_dtype(df[c]):
            if pd.to_numeric(df[c], errors="coerce").notna().mean() < 0.8:
                continue
        if latlon == "lat":
            v = pd.to_numeric(df[c], errors="coerce")
            if not (v.between(-90, 90).mean() > 0.9):
                continue
        if latlon == "lon":
            v = pd.to_numeric(df[c], errors="coerce")
            if not (v.between(-180, 180).mean() > 0.9):
                continue
        return c
    return None


def detect_schema(df: pd.DataFrame, override: dict) -> dict:
    cols = list(df.columns)
    s = {
        "datetime": _first_match(cols, DTYPE_PAT),
        "date": _first_match(cols, DATE_PAT),
        "time": _first_match(cols, TIME_PAT),
        "latitude": _first_match(cols, LAT_PAT, df, numeric=True, latlon="lat"),
        "longitude": _first_match(cols, LON_PAT, df, numeric=True, latlon="lon"),
        "violation_type": _first_match(cols, VIO_PAT, df, prefer_text=True),
        "location": _first_match(cols, LOC_PAT, df, prefer_text=True),
        "zone": _first_match(cols, ZONE_PAT, df, prefer_text=True),
        "vehicle_type": _first_match(cols, VEH_PAT, df, prefer_text=True),
        "fine_amount": _first_match(cols, FINE_PAT, df, numeric=True),
    }
    # apply user overrides (non-None values win)
    for k, v in (override or {}).items():
        if v is not None:
            s[k] = v
    return s


def build_timestamp(df: pd.DataFrame, s: dict, local_tz=None) -> pd.DataFrame:
    """Create a single 'ts' datetime column from whatever time fields exist.
    If timestamps are timezone-aware and local_tz is set, convert to local time
    (critical: rush-hour analysis must be in local time, not UTC)."""
    if s.get("datetime"):
        ts = pd.to_datetime(df[s["datetime"]], errors="coerce", utc=True)
    elif s.get("date") and s.get("time"):
        ts = pd.to_datetime(df[s["date"]].astype(str) + " " + df[s["time"]].astype(str),
                            errors="coerce", dayfirst=True)
    elif s.get("date"):
        ts = pd.to_datetime(df[s["date"]], errors="coerce", dayfirst=True)
    else:
        raise ValueError("No usable date/time column found. Set 'datetime' or "
                         "'date'(+'time') in config.COLUMN_MAP.")
    # timezone handling
    if getattr(ts.dt, "tz", None) is not None:
        if local_tz:
            ts = ts.dt.tz_convert(local_tz)
        ts = ts.dt.tz_localize(None)
    df = df.copy()
    df["ts"] = ts
    n_bad = df["ts"].isna().sum()
    if n_bad:
        warnings.warn(f"Dropped {n_bad:,} rows with unparseable timestamps.")
    return df.dropna(subset=["ts"]).reset_index(drop=True)


def filter_parking(df: pd.DataFrame, s: dict, keywords) -> pd.DataFrame:
    col = s.get("violation_type")
    if not col:
        warnings.warn("No violation_type column — keeping ALL rows as 'parking-related'.")
        return df
    text = df[col].astype(str).str.lower()
    mask = pd.Series(False, index=df.index)
    for kw in keywords:
        mask |= text.str.contains(re.escape(kw.lower()), na=False)
    if mask.sum() == 0:
        warnings.warn("Parking filter matched 0 rows — wording differs. Keeping ALL "
                      "rows. Inspect the violation vocabulary and update "
                      "config.PARKING_KEYWORDS.")
        return df
    return df[mask].reset_index(drop=True)


def report_schema(s: dict, df: pd.DataFrame):
    print("\n" + "=" * 64)
    print("DETECTED SCHEMA  (override any wrong guess in config.COLUMN_MAP)")
    print("=" * 64)
    for k, v in s.items():
        print(f"  {k:<16}->  {v}")
    has_geo = bool(s.get("latitude") and s.get("longitude"))
    print(f"\n  Spatial mode : {'lat/lon present (hotspot mining)' if has_geo else 'location/zone names'}")
    if s.get("violation_type"):
        top = df[s['violation_type']].astype(str).value_counts().head(8)
        print("\n  Top violation types in file:")
        for name, c in top.items():
            print(f"    {c:>7,}  {name}")
    print("=" * 64 + "\n")


def load(path, column_map, parking_keywords, verbose=True, local_tz=None):
    """Full load: read -> detect -> timestamp(+tz) -> parking filter."""
    df = load_raw(path)
    s = detect_schema(df, column_map)
    if verbose:
        report_schema(s, df)
    df = build_timestamp(df, s, local_tz=local_tz)
    n_before = len(df)
    df = filter_parking(df, s, parking_keywords)
    if verbose:
        print(f"Parking-related rows: {len(df):,} / {n_before:,} "
              f"({100*len(df)/max(n_before,1):.1f}%)\n")
    return df, s
