"""
Generates a realistic synthetic traffic-police violation dataset so ParkIQ can be
demoed and tested end-to-end WITHOUT the real file. The real dataset replaces this.

It bakes in the structure a good model should recover:
  - a handful of CHRONIC parking hotspots near "commercial / metro / junction" spots
  - rush-hour peaks and weekday > weekend effects
  - a long tail of sporadic one-off violations (noise)
  - mixed violation types (parking + non-parking) so the parking filter has work to do
"""
import numpy as np
import pandas as pd

# Bengaluru-ish bounding box (Flipkart HQ city — keeps the demo on-theme).
CITY_LAT, CITY_LON = 12.9716, 77.5946

CHRONIC_HOTSPOTS = [
    # (name, lat, lon, daily_lambda, rush_bias)
    ("MG Road Metro Junction",        12.9758, 77.6096, 9.0, 0.75),
    ("Koramangala 80ft Main Road",    12.9352, 77.6245, 7.5, 0.70),
    ("Marathahalli Market Circle",    12.9591, 77.6974, 6.5, 0.65),
    ("Indiranagar 100ft Commercial",  12.9719, 77.6412, 6.0, 0.72),
    ("Majestic Bus Station Signal",   12.9767, 77.5713, 8.0, 0.60),
    ("Whitefield Mall Cross",         12.9698, 77.7499, 5.0, 0.68),
]

PARKING_TYPES = [
    "Parking in No Parking Area", "Obstructive Parking",
    "Parking on Footpath", "Double Parking", "Stopping on Carriageway",
]
OTHER_TYPES = [
    "Signal Jump", "No Helmet", "Over Speeding", "Triple Riding",
    "Drunken Driving", "Using Mobile While Driving",
]
VEHICLES = ["Car", "Two Wheeler", "Auto Rickshaw", "LMV", "Truck", "Bus", "Taxi"]


def _sample_times(n, rush_bias, rng):
    """Hours skewed toward rush windows by rush_bias."""
    rush = np.concatenate([np.arange(8, 11), np.arange(17, 21)])
    flat = np.arange(0, 24)
    is_rush = rng.random(n) < rush_bias
    hours = np.where(is_rush, rng.choice(rush, n), rng.choice(flat, n))
    minutes = rng.integers(0, 60, n)
    return hours, minutes


def generate(n_days=151, seed=42) -> pd.DataFrame:  # Jan 1 -> May 31 ≈ 151 days
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    rows = []

    # --- chronic hotspots --------------------------------------------------
    for name, lat, lon, lam, rush in CHRONIC_HOTSPOTS:
        for d in range(n_days):
            day = start + pd.Timedelta(days=d)
            wk = 1.0 if day.dayofweek < 5 else 0.55          # weekday effect
            count = rng.poisson(lam * wk)
            if count == 0:
                continue
            hrs, mins = _sample_times(count, rush, rng)
            for h, m in zip(hrs, mins):
                rows.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "time": f"{h:02d}:{m:02d}",
                    "latitude": lat + rng.normal(0, 0.0007),
                    "longitude": lon + rng.normal(0, 0.0007),
                    "location": name,
                    "violation_type": rng.choice(PARKING_TYPES, p=[.4, .25, .15, .12, .08]),
                    "vehicle_type": rng.choice(VEHICLES),
                    "fine_amount": int(rng.choice([200, 500, 1000, 1500])),
                    "zone": name.split()[0],
                })

    # --- sporadic background noise (parking + other), scattered city-wide ---
    n_noise = 9000
    for _ in range(n_noise):
        day = start + pd.Timedelta(days=int(rng.integers(0, n_days)))
        h = int(rng.integers(0, 24)); m = int(rng.integers(0, 60))
        is_park = rng.random() < 0.45
        rows.append({
            "date": day.strftime("%Y-%m-%d"),
            "time": f"{h:02d}:{m:02d}",
            "latitude": CITY_LAT + rng.normal(0, 0.05),
            "longitude": CITY_LON + rng.normal(0, 0.05),
            "location": rng.choice(["Service Road", "Residential Lane", "Outer Road",
                                     "Cross Street", "Market Lane", "Ring Road"]),
            "violation_type": (rng.choice(PARKING_TYPES) if is_park
                               else rng.choice(OTHER_TYPES)),
            "vehicle_type": rng.choice(VEHICLES),
            "fine_amount": int(rng.choice([200, 500, 1000, 1500])),
            "zone": rng.choice(["North", "South", "East", "West", "Central"]),
        })

    df = pd.DataFrame(rows)
    df.insert(0, "violation_id", np.arange(1, len(df) + 1))
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    d = generate()
    d.to_csv("data/synthetic_violations.csv", index=False)
    print(f"Wrote data/synthetic_violations.csv  ({len(d):,} rows)")
    print(d.head())
