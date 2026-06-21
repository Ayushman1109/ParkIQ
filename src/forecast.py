"""
forecast.py — proactive violation forecasting (reactive -> predictive).

Builds a (hotspot x day) panel, engineers lag / rolling / calendar features, and
trains a LightGBM model with a POISSON objective (we're predicting counts, so
Poisson is the statistically correct loss, not plain MSE).

IMPORTANT on "accuracy":
  This is a prioritisation problem, not a classification problem. The question a
  patrol commander asks is "which spots will be worst *tomorrow*", so the right
  metrics are RANKING metrics on the held-out future:
      Precision@K  — of the K spots we flagged, how many were truly in the
                     real top-K the next period
      NDCG@K       — rank-quality, rewarding getting the worst spots near the top
  We also report MAE / RMSE / Poisson-deviance for the count fit.
  Validation is a strict TIME split (train on the past, test on the future) — never
  random K-fold, which would leak tomorrow into today.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb


def build_panel(df, bin_freq="D"):
    """Dense (hotspot x time-bin) panel of violation counts."""
    work = df[df["hotspot"].astype(str) != "-1"].copy()
    work["tbin"] = pd.to_datetime(work["ts"]).dt.floor(bin_freq if bin_freq != "D" else "D")
    panel = (work.groupby(["hotspot", "tbin"]).size().rename("y").reset_index())

    # make it dense: every hotspot x every bin in range (fill gaps with 0)
    bins = pd.date_range(panel["tbin"].min(), panel["tbin"].max(),
                         freq=bin_freq)
    hs = panel["hotspot"].unique()
    full = pd.MultiIndex.from_product([hs, bins], names=["hotspot", "tbin"])
    panel = (panel.set_index(["hotspot", "tbin"]).reindex(full, fill_value=0)
             .reset_index())
    return panel


def add_features(panel):
    p = panel.sort_values(["hotspot", "tbin"]).copy()
    p["dow"] = p["tbin"].dt.dayofweek
    p["is_weekend"] = (p["dow"] >= 5).astype(int)
    p["month"] = p["tbin"].dt.month
    p["dom"] = p["tbin"].dt.day
    g = p.groupby("hotspot")["y"]
    for lag in (1, 7, 14):
        p[f"lag{lag}"] = g.shift(lag)
    for win in (7, 14, 28):
        p[f"roll{win}"] = g.shift(1).rolling(win, min_periods=1).mean().reset_index(0, drop=True)
    p["hotspot_mean"] = g.transform("mean")          # spot's overall intensity
    p["hotspot_code"] = p["hotspot"].astype("category").cat.codes
    return p.dropna(subset=["lag14"]).reset_index(drop=True)


def _ndcg_at_k(true_rank_vals, pred_rank_vals, k):
    order = np.argsort(-pred_rank_vals)[:k]
    gains = true_rank_vals[order]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = np.sum(gains * discounts)
    ideal = np.sort(true_rank_vals)[::-1][:k]
    idcg = np.sum(ideal * discounts[:len(ideal)])
    return dcg / idcg if idcg > 0 else 0.0


def train_forecast(panel_feat, test_fraction=0.2, top_k=20):
    bins = np.sort(panel_feat["tbin"].unique())
    split = bins[int(len(bins) * (1 - test_fraction))]
    train = panel_feat[panel_feat["tbin"] < split]
    test = panel_feat[panel_feat["tbin"] >= split]

    feat_cols = [c for c in panel_feat.columns
                 if c not in ("hotspot", "tbin", "y")]
    Xtr, ytr = train[feat_cols], train["y"]
    Xte, yte = test[feat_cols], test["y"]

    model = lgb.LGBMRegressor(
        objective="poisson", n_estimators=400, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr, ytr)
    pred = np.clip(model.predict(Xte), 0, None)

    # count-fit metrics
    mae = float(np.mean(np.abs(pred - yte)))
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    eps = 1e-9
    pois_dev = float(2 * np.mean(yte * np.log((yte + eps) / (pred + eps)) - (yte - pred)))

    # ranking metrics: aggregate predicted vs actual over the test window per spot
    te = test.copy(); te["pred"] = pred
    agg = te.groupby("hotspot").agg(actual=("y", "sum"), pred=("pred", "sum"))
    k = min(top_k, len(agg))
    true_top = set(agg["actual"].sort_values(ascending=False).head(k).index)
    pred_top = set(agg["pred"].sort_values(ascending=False).head(k).index)
    prec_at_k = len(true_top & pred_top) / k if k else 0.0
    ndcg = _ndcg_at_k(agg["actual"].values, agg["pred"].values, k)

    metrics = {
        "test_bins": int(len(bins) - int(len(bins) * (1 - test_fraction))),
        "MAE": round(mae, 4), "RMSE": round(rmse, 4),
        "Poisson_deviance": round(pois_dev, 4),
        f"Precision@{k}": round(prec_at_k, 4),
        f"NDCG@{k}": round(ndcg, 4),
    }
    importances = (pd.Series(model.feature_importances_, index=feat_cols)
                   .sort_values(ascending=False))
    return model, metrics, importances, feat_cols


def forecast_next(model, panel_feat, feat_cols):
    """Predict the next bin's expected count per hotspot (the deployable output)."""
    last = (panel_feat.sort_values("tbin").groupby("hotspot").tail(1).copy())
    last["pred_next"] = np.clip(model.predict(last[feat_cols]), 0, None)
    return last[["hotspot", "pred_next"]].sort_values("pred_next", ascending=False)
