# ParkIQ — Predictive Parking-Congestion Intelligence

> **Problem:** *How can AI-driven parking intelligence detect illegal parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?*

ParkIQ turns the Bengaluru Traffic Police parking-violation log into a **deployable
enforcement plan**. It is built and tuned on the real dataset:

- **298,445** parking violations, **Nov 2023 -> Apr 2024**, all geocoded in Bengaluru
- **27** offence types (WRONG PARKING, NO PARKING, PARKING IN A MAIN ROAD, ...)
- **169** official BTP junctions + **54** police stations in the data

It answers the three stated "Why it's hard today" pain points head-on:

| Their pain point | ParkIQ layer | What's novel |
|---|---|---|
| *"No heatmap of violations vs congestion impact"* | **SEE** — hotspot map + dashboard | We map *impact* (CIS), not just dot density |
| *"Difficult to prioritise enforcement zones"* | **SCORE** — Congestion Impact Score | Transparent 0-100 index, anchored to BTP junctions |
| *"Enforcement is patrol-based and reactive"* | **ACT** — forecast + patrol optimiser | Predict tomorrow, then *optimally* deploy patrols |

---

## Real results (straight from `python run_pipeline.py`)

**Top congestion-impact hotspots (CIS):**

| Rank | Junction | Violations | CIS |
|---|---|---|---|
| 1 | BTP044 - Sagar Theatre Junction | 10,549 | 89.5 |
| 2 | BTP040 - Elite Junction | 10,718 | 88.4 |
| 3 | BTP082 - KR Market Junction | 11,538 | 87.1 |
| 4 | BTP051 - Safina Plaza Junction | 15,449 | 87.0 |
| 5 | BTP080 - NR Road / SP Road Junction | 3,681 | 86.1 |

**Forecast (held-out future weeks, strict time-split):**
`Precision@20 = 0.95`, `NDCG@20 = 0.995`, `MAE ~ 4.1` violations/day per hotspot.
The model nails *which* junctions will be worst next period — the question that
actually drives deployment.

**Optimiser** emits a ready-to-hand-out patrol schedule (`outputs/patrol_schedule.csv`)
assigning your N units across morning/afternoon/evening/night shifts to the highest
expected-impact junctions.

---

## The four ideas that win the room

**1. Congestion Impact Score (CIS) — quantifying impact, not just counting.**
A spot with thousands of scattered low-severity tickets is not the same problem as
one blocking a main-road junction during the city's busy hours. CIS blends five
explainable 0-1 components into one rankable 0-100 number, and ships the breakdown
so an officer sees *why* a spot ranks high (no black box):

- **Density** — offending volume (log-scaled)
- **Recurrence** — how chronic (share of days active)
- **Peakedness** — share of offending in the city's busy hours
- **Criticality** — *real BTP junction presence* + *offence severity* (main-road,
  road-crossing, footpath, bus-stop offences score higher)
- **Spillover** — **graph diffusion** over the junction network (see below)

**2. Hotspots anchored to the police's own junction taxonomy.** Instead of arbitrary
clusters, hotspots are the official `BTP###` junctions wherever a violation carries
one; off-junction violations are grid-binned (~150 m) and merged. Output speaks the
BTP's language — instantly actionable.

**3. Reactive -> predictive.** A LightGBM model (**Poisson** objective — we predict
*counts*) forecasts each hotspot's next-period load from lag/rolling/calendar
features, validated on the **future** (never random K-fold) and scored with
**ranking** metrics (Precision@K, NDCG@K).

**4. Allocate AND route, to a coverage target.** The optimiser runs in
**coverage mode** — pick the minimum set of junctions to put eyes on a target share
(default **80%**) of all violations — then a **TSP route** (nearest-neighbour +
2-opt) per patrol per shift gives the shortest visiting order. (An `impact` mode that
maximises CIS-weighted impact with PuLP/CBC is also available.) The map shows the
routes as toggleable per-patrol layers.

---

## Graph-based spillover (the honest version of the "GNN" idea)

A natural temptation is a Spatio-Temporal Graph CNN that predicts cascading traffic
*delay*. We deliberately did **not** do that here, and that's a strength to own in
Q&A: a supervised delay model needs flow / speed / capacity ground-truth, and this
dataset has none — it is parking-violation records only. Training a delay GNN against
labels that don't exist would be indefensible the moment a judge asks "what was your
ground truth?"

Instead we use the graph *structurally*, with no invented labels: hotspots are nodes,
proximity is a Gaussian-kernel edge weight, and we diffuse violation intensity 1–2
hops (`s = (W + decay·W²)·v`). A junction ringed by other heavy junctions scores high
spillover; an isolated one scores low — an unsupervised measure of cascading risk.
This is why **Elite Junction** (deep in a dense CBD cluster) outranks the
higher-volume but more **isolated Safina Plaza** in the final CIS. Drop in a real
speed/flow feed later and this same graph becomes the substrate for a *supervised*
STGCN — but only once the labels exist.

---

## Real-data engineering wins (great "we went deep" slides)

- **Timezone correctness.** `created_datetime` is **UTC** (`+00`). Naively reading it
  would put Bengaluru's activity in the wrong hours. ParkIQ converts to **IST**
  before any time-of-day analysis.
- **Data-driven peak hours.** The timestamp is *challan-creation* time and is
  morning-shift heavy (peaks 09:00-12:00 IST, not evening commute). So ParkIQ defines
  "peak" from the data's own busy hours rather than assuming 5-8pm — robust to the
  timestamp's real meaning.
- **Scalability.** Classic HDBSCAN OOMs on 298k points; ParkIQ uses a fast
  memory-safe grid + junction anchoring instead (runs in seconds).

---

## Why our "accuracy" story is more sophisticated than a single number

This is a **prioritisation** problem, not classification. Optimising raw accuracy
would be the wrong target. We predict **counts** with a Poisson loss, validate on the
**future**, and report **Precision@K / NDCG@K** — "of the spots we flagged, how many
were genuinely worst next period?" On the real data: **0.95 / 0.995**. Lead with that;
MAE/RMSE are supporting detail.

---

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py              # interactive demo dashboard (uses outputs/)
```

### The demo dashboard (`app.py`)

A Streamlit prototype built to *survive a live demo*:

1. **AI Dispatcher ("Executive Copilot")** - a natural-language query box that is
   **deterministic intent-parsing, not a live LLM**, so it cannot hallucinate and needs
   no network/API key (zero live-demo failure surface). It is **robust to free-form
   English** — synonyms ("worst / chokepoint / problem"), number-words ("worst three"),
   **fuzzy** division/junction/area matching (typos and partial names like "koramngala"
   resolve), and a **smart fallback that never dead-ends** (an unrecognised question
   still returns the top hotspots). It handles **compound** queries: *"plan a patrol for
   Koramangala"* -> finds that area's hotspots and computes a live 2-opt route; *"where's
   the worst parking problem?"*, *"top 5 chokepoints with a route"*, *"BTP082"*,
   *"morning route"*. A focused junction also shows a **CIS component radar** explaining
   *why* it scores what it does. (An LLM can map English -> the same structured intents
   behind this interface if a stable network is available — the deterministic parser is
   what survives a demo, and every number still comes from the data, never the model.)
   - **💰 Business case panel** translates the routing saving into money & time:
     editable assumptions (km/l, ₹/l, city speed) -> litres, ₹/month, and **policing
     hours returned** (e.g. ~₹38k/month and ~161 hrs/month on the demo fleet).
   - **🚧 What-if disruption** simulator: pick a junction, scale its congestion impact
     (metro works / closure / event), and watch CIS **re-rank** it and the optimiser
     **pull it into the patrol plan** and re-sequence the route — proving the model
     adapts, not just reports history. (Re-routing = re-selection + re-sequencing on
     straight-line distance; no road-graph avoidance, stated honestly.)
2. **Before/After diffusion map** - a 3D `pydeck` map toggling **Raw Counts** vs
   **AI Diffused Impact (CIS)**, with a "biggest movers" panel (Elite Junction climbs
   to #1, Safina Plaza falls) - visual proof the graph term earns its place.
3. **Coverage Planner** - a **live slider** plus an **"optimise for" toggle**:
   maximise coverage of either raw **violations** or **congestion impact** (each
   violation severity-weighted 0.5x–2.0x by its hotspot's arterial/junction criticality
   x rush-hour concentration — directly answering the prompt's *"quantify impact on
   traffic flow"*). Drag the target and watch junction count, coverage, and the cost
   **curve** update live, with the efficiency **knee** ("sweet spot — past here is wasted
   budget") marked. Impact mode is *more* efficient: ~80% of congestion impact needs ~90
   junctions vs ~159 for raw volume. A **beat-radius** control runs greedy
   Maximum-Coverage to trade stop-count for beat-size. Honest denominator (all
   violations): 80% of violations needs ~159 junctions; ceiling ~88%.
4. **Enforcement-bias view** - raw challan volume vs an **explicitly assumption-based**
   demand estimate, flags the enforcement-gap hours (~15:00-21:00), and turns it into
   an action. We label the estimate as an estimate - never "reality".
5. **Equity check** - coverage broken down by **police division**, exposing that
   efficiency-first deployment over-serves the dense core and leaves outer divisions
   under-served (7 at 0% coverage). A **fairness-floor toggle** guarantees >=1 patrolled
   hotspot per division - live, it eliminates all zero-coverage divisions and improves
   the coverage **Gini** for ~9 extra junctions and ~0.5% citywide coverage. A
   responsible-deployment angle most teams never show.
6. **Operational ROI** - `st.metric` strip with **real, computed** numbers: violation
   coverage, TSP-vs-naive distance saved, and **patrol efficiency** (violations
   covered per route-km).

**Demo-survivability:** if `outputs/` is missing, `app.py` falls back to
`mock_data.py` (embedding the real top-junction values), so it renders on any machine.

Outputs land in `outputs/`:

- `cis_ranked_hotspots.csv` — ranked hotspots with full CIS breakdown
- `patrol_schedule.csv` — the optimiser's shift-by-shift deployment plan
- `patrol_routes.csv` — TSP visiting order + km per patrol per shift
- `coverage_curve.csv` — coverage % vs # junctions (the diminishing-returns curve)
- `temporal_profile.csv` — hourly/weekday violation counts (for the bias view)
- `equity_zone_hotspot.csv` — violations per division × hotspot (for the equity view)
- `kepler_timelapse.csv` — hotspot points by **month**, ready to drop into
  [kepler.gl](https://kepler.gl) for a dark-mode 3D time-lapse (animate by `month`,
  **not** hour — hourly reflects challan-creation bias, not real demand)
- `hotspot_map.html` — Bengaluru heatmap + CIS markers + toggleable patrol routes
- `charts.html` — hour x weekday matrix, monthly trend, CIS leaderboard & breakdown
- `summary.json` — machine-readable run summary

If the data file is missing, ParkIQ auto-generates a synthetic dataset so the
pipeline always runs for a demo.

## Reproducing the results (for judges)

The real outputs are already in `outputs/` and the app reads them, so
`streamlit run app.py` shows the real Bengaluru results **with no data file**.

To re-run the full pipeline from the source data and reproduce the numbers:
drop the official dataset into `data/` (any of `.csv`, `.xlsb`, `.xlsx`, `.parquet`
— the 26 MB `.xlsb` and the larger original `.csv` are the same 298,445 rows and give
identical results), then `python run_pipeline.py`. ParkIQ **auto-detects** whatever
real file is in `data/` (no config edit needed). With no data file present it falls
back to a synthetic demo dataset (clearly flagged; numbers are illustrative, not real).

## Adapting / re-tuning (all in `config.py`)

- `DATA_PATH` — the dataset (`.xlsb/.xlsx/.csv/.parquet` all supported)
- `COLUMN_MAP` — already mapped to this schema; auto-detects if you blank a field
- `LOCAL_TZ` — timezone to convert into (default `Asia/Kolkata`)
- `SPATIAL_METHOD` — `junction` (default) / `grid` / `hdbscan` / `names`
- `CIS_WEIGHTS`, `OFFENCE_SEVERITY`, `PEAK_MODE`, `PATROL_UNITS`, `SHIFTS`
- `OPTIMIZE_MODE` (`coverage`/`impact`) and `COVERAGE_TARGET` (e.g. `0.80`)

## Architecture

```
config.py            <- all tunables (data path, mappings, weights, shifts)
run_pipeline.py      <- orchestrates the 6 stages, writes outputs/
src/
  loader.py          <- multi-format load + schema detect + UTC->IST + parking filter
  features.py        <- temporal + data-driven peaks + offence severity
                        + junction-anchored / grid hotspot mining
  cis.py             <- Congestion Impact Score
  forecast.py        <- LightGBM Poisson forecast + time-split + ranking metrics
  optimizer.py       <- ILP patrol allocation (PuLP) + greedy fallback
  routing.py         <- TSP visiting route per shift (nearest-neighbour + 2-opt)
  dashboard.py       <- Folium map (+ route layers) + Plotly dashboard
synthetic_data.py    <- demo data fallback for the pipeline
app.py               <- Streamlit demo (4 modules: dispatcher, diffusion, bias, ROI)
app_logic.py         <- pure, unit-tested logic behind the app (no UI)
mock_data.py         <- realistic stand-in CSVs so the app runs with no pipeline
```

## Honest limitations -> your "future work" slide

- **Travel cost is straight-line (haversine), not road-network distance.** Routes use
  great-circle distance between stops — fine and fast for a prototype, and a
  conservative (slightly optimistic) approximation. To use real driving distance, swap
  the distance matrix in `src/routing.py` for an OSRM / Mapbox / Google Distance-Matrix
  call; the TSP/2-opt step is unchanged.
- **Coverage is bounded by spatial spread, not algorithm.** Reaching 90% needs ~150
  zones because violations are spread across the city (the top 12 hotspots are only
  ~32% of all violations). The Coverage Planner's **beat-radius** control runs greedy
  Maximum-Coverage to trade stop-count for beat-size (150 m→336 zones, 1500 m→77 zones
  for ~90%) — a deliberate operational choice, not a way to fake a small number.

- **No live traffic feed.** CIS uses violation structure + junction + offence
  severity as the flow-impact signal. Drop a Google/TomTom/Mapbox speed feed into
  `src/cis.py` and CIS becomes a *measured* flow impact rather than a strong proxy.
- **`created_datetime` is challan-creation time**, not necessarily the moment of
  offence — hence the data-driven peak approach. A true occurrence-time field would
  sharpen the temporal model further.

## Suggested 3-minute pitch arc

1. *The gap* — enforcement is reactive and blind to impact (their words).
2. *SEE* — the Bengaluru heatmap... but density != congestion impact.
3. *SCORE* — CIS ranks Sagar Theatre / Elite / KR Market junctions, and shows *why*.
4. *ACT* — we forecast next week (Precision@20 = 0.95) and hand the commander this
   optimised patrol roster.
5. *Depth* — UTC->IST fix, data-driven peaks, junction anchoring, scalable to 298k.
   One config line points it at next month's data.
