"""
ParkIQ — central configuration.

THE ONLY FILE YOU NEED TO TOUCH to plug in the real Flipkart/HackerEarth dataset.

Workflow:
  1. Run once with COLUMN_MAP = {} (all None). ParkIQ auto-detects the schema and
     prints what it found.
  2. If auto-detection got anything wrong, hard-code the correct column names below.
  3. Re-run. Done.
"""

# ---------------------------------------------------------------------------
# 1. WHERE IS THE DATA
# ---------------------------------------------------------------------------
# Reads .xlsb / .xlsx / .csv / .parquet directly (engine picked by extension).
# Auto-falls back to a generated synthetic dataset if the file is missing.
DATA_PATH = "data/jan_to_may_police_violation_anonymized791b166__1_.xlsb"

OUTPUT_DIR = "outputs"

# Timestamps in this file are timezone-aware UTC ("+00"). Bengaluru is IST, and
# rush-hour analysis MUST be in local time, so we convert. (Set LOCAL_TZ=None to
# leave timestamps untouched if a future file is already local/naive.)
LOCAL_TZ = "Asia/Kolkata"

# ---------------------------------------------------------------------------
# 2. COLUMN MAP  (mapped to the real Bengaluru Traffic Police schema)
#    Leave a value as None to let ParkIQ auto-detect it.
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "datetime":       "created_datetime",
    "date":           None,
    "time":           None,
    "latitude":       "latitude",
    "longitude":      "longitude",
    "violation_type": "violation_type",   # JSON-style list e.g. ["WRONG PARKING"]
    "location":       "location",
    "zone":           "police_station",
    "vehicle_type":   "vehicle_type",
    "fine_amount":    None,                # no fine column in this dataset
    "junction":       "junction_name",     # REAL criticality signal
}

# ---------------------------------------------------------------------------
# 3. PARKING FILTER
# ---------------------------------------------------------------------------
# We only care about parking-induced congestion, so we keep rows whose
# violation-type text contains any of these (case-insensitive substring match).
# Tune after you see the real violation_type vocabulary printed by the profiler.
PARKING_KEYWORDS = [
    "parking", "no parking", "no-parking", "obstruct", "footpath",
    "pavement", "double park", "wrong side", "stopping", "halt",
    "tow", "abandon",
]
# If NONE of the rows match (e.g. wording is totally different), ParkIQ keeps all
# rows and warns you, so the pipeline never silently dies.

# ---------------------------------------------------------------------------
# 4. SPATIAL UNIT
# ---------------------------------------------------------------------------
# "auto"   -> grid for large geo data (>SPATIAL_HDBSCAN_MAX rows), else HDBSCAN
# "grid"   -> fast, memory-safe ~GRID_SIZE_M cells (recommended for this 298k-row file)
# "hdbscan"-> organic clusters (great for <~50k points; OOMs on very large sets)
# "names"  -> use the location/zone string as the unit (no lat/lon needed)
SPATIAL_METHOD = "junction"
GRID_SIZE_M = 150           # grid cell size in metres (~one road segment)
SPATIAL_HDBSCAN_MAX = 60000 # above this many points, "auto" avoids HDBSCAN (OOM)

# A grid cell / cluster counts as a hotspot only if it has at least this many
# violations (filters out sparse one-off spots).
MIN_CLUSTER_SIZE = 60
HOTSPOT_RADIUS_M = 120      # HDBSCAN-only: merge points within this distance

# Rush-hour windows (LOCAL time, 24h) used for the "peakedness" CIS component.
RUSH_HOURS = list(range(8, 11)) + list(range(17, 21))  # 8-10am, 5-8pm
# "data_driven" -> peak = hours with above-average volume (robust; recommended).
# "fixed"       -> use RUSH_HOURS above (classic commute assumption).
PEAK_MODE = "data_driven"

# ---------------------------------------------------------------------------
# 4b. OFFENCE SEVERITY  (data-driven criticality)
# ---------------------------------------------------------------------------
# Some parking offences hurt traffic flow / safety far more than a generic
# "no parking". This maps the real offence vocabulary -> a 0-1 severity weight
# that feeds the CIS criticality component. Unlisted offences default to 0.4.
OFFENCE_SEVERITY = {
    "PARKING NEAR ROAD CROSSING": 1.0,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 1.0,
    "PARKING IN A MAIN ROAD": 0.9,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.9,
    "DOUBLE PARKING": 0.85,
    "PARKING ON FOOTPATH": 0.8,
    "PARKING OTHER THAN BUS STOP": 0.7,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.65,
    "WRONG PARKING": 0.5,
    "NO PARKING": 0.45,
}
OFFENCE_SEVERITY_DEFAULT = 0.4

# Words in a junction/location name that confirm a congestion-critical road.
# (Now mostly a backstop — real junction_name presence drives criticality.)
CRITICAL_KEYWORDS = [
    "junction", "circle", "cross", "signal", "flyover", "metro", "station",
    "market", "mall", "main road", "ring road", "bus", "hospital", "school",
    "temple", "complex", "bazaar", "commercial",
]

# ---------------------------------------------------------------------------
# 5. CONGESTION IMPACT SCORE — component weights (must sum to 1.0)
# ---------------------------------------------------------------------------
CIS_WEIGHTS = {
    "density":     0.30,   # how many violations
    "recurrence":  0.25,   # how chronic (distinct active days) — one-off vs habitual
    "peakedness":  0.20,   # how concentrated in rush hours (= worst for flow)
    "criticality": 0.15,   # how important the road segment is
    "spillover":   0.10,   # how much it clusters with neighbouring offending
}

# ---------------------------------------------------------------------------
# 6. FORECASTING
# ---------------------------------------------------------------------------
FORECAST_TIME_BIN = "D"      # 'D' daily, 'H' hourly (needs enough volume)
TEST_FRACTION = 0.2          # last 20% of the timeline is the hold-out (TIME-based)
TOP_K = 20                   # we report Precision@K / NDCG@K for the top-K hotspots

# ---------------------------------------------------------------------------
# 7. ENFORCEMENT OPTIMIZER
# ---------------------------------------------------------------------------
PATROL_UNITS = 8             # how many patrol teams you can deploy per shift
SHIFTS = ["morning", "afternoon", "evening", "night"]

# Deployment objective:
#   "coverage" -> patrol the MINIMUM set of junctions to put eyes on COVERAGE_TARGET
#                 of all violations (patrols run multi-stop routes). Recommended:
#                 directly answers "what % of the problem do we cover?".
#   "impact"   -> classic: maximise CIS-weighted impact with one stop per patrol-shift.
OPTIMIZE_MODE = "coverage"
COVERAGE_TARGET = 0.80       # 0.80 = put eyes on 80% of all recorded violations
SHIFT_HOURS = {
    "morning":   list(range(6, 12)),
    "afternoon": list(range(12, 17)),
    "evening":   list(range(17, 22)),
    "night":     list(range(22, 24)) + list(range(0, 6)),
}

# Optional patrol depot (lat, lon) where routes start/end. None -> start at the
# highest-impact stop in each shift. (e.g. a control room / police HQ)
DEPOT_LATLON = None
