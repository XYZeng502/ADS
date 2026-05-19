# Experiment Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract shared core module from spend training script, then implement exp_01 quantile regression experiment with isolated output and comparison reporting.

**Architecture:** Extract `_build_model`, `_tree_predict`, `_evaluate`, `_run`, and related helpers from `scripts/run_parallel_models_spend_t1.py` into `app/experiments/core.py`. The original script re-imports from core.py with zero behavior change. `exp_01_quantile.py` imports only what it needs from core.py and overrides the model-building logic to produce P10/P50/P90 variants.

**Tech Stack:** Python 3, XGBoost, scikit-learn, pandas, numpy

---

### Task 1: Create shared experiments module

**Files:**
- Create: `app/experiments/__init__.py`
- Create: `app/experiments/core.py`
- Modify: `scripts/run_parallel_models_spend_t1.py:1-920`

**Goal:** Move core training/evaluation functions to a shared module so experiment scripts can import them without duplicating code. The original script must produce identical output after the refactor.

- [ ] **Step 1: Create `app/experiments/__init__.py`**

```bash
mkdir -p app/experiments
```

```python
# app/experiments/__init__.py
```

- [ ] **Step 2: Create `app/experiments/core.py` with extracted functions**

```python
# app/experiments/core.py
"""Shared core functions for model training and evaluation.

Extracted from scripts/run_parallel_models_spend_t1.py so experiment scripts
can import and reuse the walk-forward training pipeline.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


@dataclass
class ModelResult:
    model_name: str
    metrics: Dict[str, float]
    prediction_df: pd.DataFrame


BASE_FEATURE_COLS = [
    "消耗金额", "roi_d1", "ctr", "cvr_dl", "cvr_act",
    "dow", "is_fri_sat", "is_holiday", "is_rest_day",
    "dom", "month", "week_of_month",
    "days_to_month_end", "month_progress",
    "is_month_start_3d", "is_month_end_3d", "is_month_end_7d", "is_month_end_10d",
    "days_to_next_holiday", "days_since_prev_holiday",
    "is_pre_holiday_3d", "is_post_holiday_3d",
    "is_summer_winter_break",
    "holiday_id", "holiday_seq_index", "holiday_days_remaining", "holiday_window_len",
    "is_last_holiday_day",
    "target_dow", "target_is_weekend", "target_is_rest_day", "target_is_adjusted_workday",
    "target_is_holiday", "target_holiday_id", "target_holiday_seq_index",
    "target_holiday_days_remaining", "target_holiday_window_len",
    "target_is_last_holiday_day",
    "target_days_to_next_holiday", "target_days_since_prev_holiday",
    "target_is_pre_holiday_3d", "target_is_post_holiday_1d", "target_is_post_holiday_3d",
    "target_is_first_workday_after_holiday",
    "spend_lag_1", "spend_lag_2", "spend_lag_3", "spend_lag_7",
    "spend_roll_mean_3", "spend_roll_std_3", "spend_roll_mean_7", "spend_roll_std_7",
    "roi_lag_1", "roi_lag_2", "roi_lag_3", "roi_lag_7",
    "spend_ratio_1d",
    "traffic_n_unique", "traffic_top1_pct", "traffic_hhi",
    "scene_n_unique", "scene_top1_pct", "scene_hhi",
    "creative_n_unique", "creative_top1_pct", "creative_hhi",
    "billing_n_unique", "billing_top1_pct", "billing_hhi",
    "spend_x_is_rest_day", "spend_lag1_x_target_rest",
    "spend_lag1_x_target_holiday", "spend_ratio_x_target_rest",
    "app_label",
]

CALENDAR_PACING_FEATURE_COLS = [
    "week_of_month", "days_to_month_end", "month_progress",
    "is_month_start_3d", "is_month_end_3d", "is_month_end_7d", "is_month_end_10d",
    "days_to_next_holiday", "days_since_prev_holiday",
    "is_pre_holiday_3d", "is_post_holiday_3d", "is_summer_winter_break",
    "holiday_id", "holiday_seq_index", "holiday_days_remaining", "holiday_window_len",
    "is_last_holiday_day",
    "target_dow", "target_is_weekend", "target_is_rest_day", "target_is_adjusted_workday",
    "target_is_holiday", "target_holiday_id", "target_holiday_seq_index",
    "target_holiday_days_remaining", "target_holiday_window_len",
    "target_is_last_holiday_day",
    "target_days_to_next_holiday", "target_days_since_prev_holiday",
    "target_is_pre_holiday_3d", "target_is_post_holiday_1d", "target_is_post_holiday_3d",
    "target_is_first_workday_after_holiday",
    "spend_roll_mean_3", "spend_roll_std_3", "spend_roll_mean_7", "spend_roll_std_7",
    "mtd_spend", "mtd_revenue", "mtd_roi_d1",
    "spend_x_is_rest_day", "spend_lag1_x_target_rest",
    "spend_lag1_x_target_holiday", "spend_ratio_x_target_rest",
    "app_label",
]


def feature_cols(feature_set: str = "base") -> List[str]:
    if feature_set == "calendar_pacing":
        return list(dict.fromkeys(BASE_FEATURE_COLS + CALENDAR_PACING_FEATURE_COLS))
    return BASE_FEATURE_COLS


def safe_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_")


def build_model(name: str, use_gpu: bool = False, **overrides):
    """Build a tree model by name. Extra kwargs override defaults (e.g. quantile_alpha)."""
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
        )
    if name == "GBDT":
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.04, max_depth=3, random_state=42
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=500, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
        )
    if name == "LightGBM":
        if LGBMRegressor is None:
            return None
        kwargs = dict(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1,
        )
        if use_gpu:
            kwargs["device"] = "cuda"
        kwargs.update(overrides)
        return LGBMRegressor(**kwargs)
    if name == "XGBoost":
        if XGBRegressor is None:
            return None
        kwargs = dict(
            n_estimators=500, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=42,
            n_jobs=4, enable_categorical=True,
        )
        if use_gpu:
            kwargs["device"] = "cuda"
        kwargs.update(overrides)
        return XGBRegressor(**kwargs)
    if name == "CatBoost":
        if CatBoostRegressor is None:
            return None
        kwargs = dict(
            iterations=500, learning_rate=0.03, depth=6,
            random_seed=42, verbose=0, thread_count=4,
        )
        kwargs.update(overrides)
        return CatBoostRegressor(**kwargs)
    return None


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > 1e-8
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
        if np.any(mask) else np.nan
    )
    return {"mae": mae, "rmse": rmse, "mape": mape, "mape_pct": mape * 100.0 if np.isfinite(mape) else np.nan}


def evaluate_pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball loss for a single quantile level."""
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, alpha * errors, (alpha - 1) * errors)))


def compute_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of y_true falling within [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def compute_interval_width(lower: np.ndarray, upper: np.ndarray, median: np.ndarray) -> np.ndarray:
    """Normalized interval width: (upper - lower) / median. Clip median to avoid div-by-zero."""
    safe_median = np.where(median > 1e-8, median, 1.0)
    return (upper - lower) / safe_median


def apply_holiday_transition_calibration(test_df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(pred, dtype=float).copy()
    current_spend = pd.to_numeric(test_df["消耗金额"], errors="coerce").fillna(0.0).to_numpy()
    roll3 = pd.to_numeric(test_df.get("spend_roll_mean_3", 0.0), errors="coerce").fillna(0.0).to_numpy()
    roll7 = pd.to_numeric(test_df.get("spend_roll_mean_7", 0.0), errors="coerce").fillna(0.0).to_numpy()
    recent_ref = np.maximum(current_spend, roll3)
    wider_ref = np.maximum(recent_ref, roll7)

    last_holiday_day = pd.to_numeric(test_df.get("target_is_last_holiday_day", 0), errors="coerce").fillna(0).to_numpy() == 1
    first_workday_after_holiday = (
        pd.to_numeric(test_df.get("target_is_first_workday_after_holiday", 0), errors="coerce").fillna(0).to_numpy() == 1
    )

    adjusted[last_holiday_day] = np.maximum(adjusted[last_holiday_day], recent_ref[last_holiday_day] * 0.80)
    adjusted[first_workday_after_holiday] = np.minimum(
        adjusted[first_workday_after_holiday],
        recent_ref[first_workday_after_holiday] * 0.45,
    )

    target_is_holiday = pd.to_numeric(test_df.get("target_is_holiday", 0), errors="coerce").fillna(0).to_numpy() == 1
    target_is_weekend = pd.to_numeric(test_df.get("target_is_weekend", 0), errors="coerce").fillna(0).to_numpy() == 1
    mid_holiday_weekday = target_is_holiday & ~target_is_weekend & ~last_holiday_day & ~first_workday_after_holiday
    adjusted[mid_holiday_weekday] = np.maximum(adjusted[mid_holiday_weekday], wider_ref[mid_holiday_weekday] * 0.90)

    target_is_rest_day = pd.to_numeric(test_df.get("target_is_rest_day", 0), errors="coerce").fillna(0).to_numpy() == 1
    days_since_holiday = pd.to_numeric(test_df.get("target_days_since_prev_holiday", 99), errors="coerce").fillna(99).to_numpy()

    early_ph_rest = target_is_rest_day & ~target_is_holiday & (days_since_holiday >= 3) & (days_since_holiday <= 4)
    adjusted[early_ph_rest] = np.maximum(adjusted[early_ph_rest], wider_ref[early_ph_rest] * 0.90)
    adjusted[early_ph_rest] = np.minimum(adjusted[early_ph_rest], wider_ref[early_ph_rest] * 1.35)

    late_ph_rest = target_is_rest_day & ~target_is_holiday & (days_since_holiday >= 5) & (days_since_holiday <= 7)
    adjusted[late_ph_rest] = np.maximum(adjusted[late_ph_rest], wider_ref[late_ph_rest] * 0.80)
    adjusted[late_ph_rest] = np.minimum(adjusted[late_ph_rest], wider_ref[late_ph_rest] * 1.10)

    return np.clip(adjusted, 0.0, None)


def tree_predict(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 feats: List[str], use_log_target: bool, use_gpu: bool = False,
                 weight_exponent: float = 0.35, model_overrides: Optional[Dict] = None) -> pd.DataFrame:
    """Train a single tree model and return predictions.

    Args:
        name: Model name (XGBoost, CatBoost, LightGBM, etc.)
        model_overrides: Extra kwargs passed to build_model() to override defaults.
                         Used by quantile experiments to set objective/quantile_alpha.
    """
    train = train_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    model = build_model(name, use_gpu=use_gpu, **(model_overrides or {}))
    if model is None:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    x_train = train[feats].copy()
    y_train = train["target_t1_spend"].clip(lower=0.0)
    x_test = test[feats].copy()
    y_test = test["target_t1_spend"].values
    label = name
    if use_log_target:
        y_train = np.log1p(y_train)
        label = f"{name}_log"

    cat_cols = [c for c in ["app_label"] if c in x_train.columns]
    fit_kwargs = {}
    if name == "CatBoost" and cat_cols:
        for c in cat_cols:
            x_train[c] = x_train[c].astype("category")
            x_test[c] = x_test[c].astype("category")
        fit_kwargs["cat_features"] = cat_cols

    sample_weight = np.power(train["消耗金额"].clip(lower=0).values, weight_exponent)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight, **fit_kwargs)
    except TypeError:
        model.fit(x_train, y_train, **fit_kwargs)
    pred = model.predict(x_test)
    if use_log_target:
        log_resid = y_train - model.predict(x_train)
        smearing = float(np.clip(np.exp(log_resid).mean(), 0.8, 1.25))
        pred = np.expm1(pred) * smearing
    pred = np.clip(pred, 0.0, None)
    pred = apply_holiday_transition_calibration(test, pred)
    return pd.DataFrame({
        "日期": test["日期"].values, "应用ID": test["应用ID"].values,
        "y_true": y_test, "y_pred": pred, "model": label,
    })


def ewma_spend(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    preds = []
    for app_id, g_test in test_df.groupby("应用ID"):
        vals = train_df[train_df["应用ID"] == app_id].sort_values("日期")["消耗金额"].dropna().values
        if len(vals) == 0:
            p = np.nan
        else:
            s = vals[0]
            for v in vals[1:]:
                s = alpha * v + (1 - alpha) * s
            p = s
        for _, r in g_test.iterrows():
            preds.append({
                "日期": r["日期"], "应用ID": r["应用ID"],
                "y_true": r["target_t1_spend"], "y_pred": p, "model": "EWMA",
            })
    return pd.DataFrame(preds)


def run(df: pd.DataFrame, min_train_days: int, alpha: float,
        use_log_target: bool, feature_set: str = "base",
        min_target_spend_train: float = 0.0, min_target_spend_eval: float = 0.0,
        *, eval_recent_days: Optional[int] = None,
        tree_models: Optional[List[str]] = None,
        skip_baseline_two_stage: bool = False, use_gpu: bool = False,
        weight_exponent: float = 0.35,
        model_overrides: Optional[Dict] = None,
        extra_model_variants: Optional[List[Dict]] = None) -> List[ModelResult]:
    """Walk-forward training loop.

    Args:
        model_overrides: Default kwargs for all tree models (e.g. quantile settings).
        extra_model_variants: List of dicts with 'suffix' and 'overrides' keys to train
                              additional model variants. E.g. [{'suffix': 'q10', 'overrides': {'objective': 'reg:quantileerror', 'quantile_alpha': 0.1}}]
    """
    feats = feature_cols(feature_set)
    models = list(tree_models) if tree_models else ["RandomForest", "GBDT", "ExtraTrees", "LightGBM"]
    preds_by_model: Dict[str, List[pd.DataFrame]] = {"EWMA": []}
    for m in models:
        preds_by_model[m] = []
        if use_log_target:
            preds_by_model[f"{m}_log"] = []
    if extra_model_variants:
        for variant in extra_model_variants:
            for m in models:
                preds_by_model[f"{m}_{variant['suffix']}"] = []
                if use_log_target:
                    preds_by_model[f"{m}_log_{variant['suffix']}"] = []

    _enc_app_avg_spend = "app_avg_spend"
    _enc_app_avg_roi = "app_avg_roi_d1"
    model_df = df.sort_values(["应用ID", "日期"]).reset_index(drop=True).copy()
    model_df["split_value"] = "ALL"
    grp_spend = model_df.groupby("应用ID")["target_t1_spend"]
    model_df[_enc_app_avg_spend] = (
        grp_spend.cumsum().sub(model_df["target_t1_spend"])
        / grp_spend.cumcount().clip(lower=1)
    )
    model_df[_enc_app_avg_spend] = model_df[_enc_app_avg_spend].fillna(
        model_df["target_t1_spend"].median()
    )
    if "roi_d1" in model_df.columns:
        grp_roi = model_df.groupby("应用ID")["roi_d1"]
        model_df[_enc_app_avg_roi] = (
            grp_roi.cumsum().sub(model_df["roi_d1"])
            / grp_spend.cumcount().clip(lower=1)
        )
        model_df[_enc_app_avg_roi] = model_df[_enc_app_avg_roi].fillna(
            model_df["roi_d1"].median()
        )
        enc_feats = [_enc_app_avg_spend, _enc_app_avg_roi]
    else:
        enc_feats = [_enc_app_avg_spend]
    feats_all = feats + enc_feats

    dates = sorted(model_df["日期"].dropna().unique())
    cutoffs = list(dates[:-1])
    if eval_recent_days is not None and eval_recent_days > 0 and len(cutoffs) > eval_recent_days:
        cutoffs = cutoffs[-eval_recent_days:]
    n_cutoffs = len(cutoffs)
    for i, cutoff in enumerate(cutoffs):
        t_cutoff = pd.Timestamp(cutoff)
        next_day = cutoff + np.timedelta64(1, "D")
        train = model_df[model_df["日期"] <= t_cutoff].copy()
        test = model_df[model_df["日期"] == next_day].copy()
        if min_target_spend_train > 0:
            train = train[train["target_t1_spend"] >= min_target_spend_train].copy()
        if min_target_spend_eval > 0:
            test = test[test["target_t1_spend"] >= min_target_spend_eval].copy()
        if test.empty or train["日期"].nunique() < min_train_days:
            continue
        print(f"  [{i+1}/{n_cutoffs}] cutoff={t_cutoff.date()} train={len(train)} test={len(test)}", flush=True)
        ew = ewma_spend(train, test, alpha)
        if not ew.empty:
            preds_by_model["EWMA"].append(ew)
        max_workers = max(2, min(8, len(models) * (2 if use_log_target else 1)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for m in models:
                futures[ex.submit(tree_predict, m, train, test, feats_all, False, use_gpu, weight_exponent, model_overrides)] = m
                if use_log_target:
                    futures[ex.submit(tree_predict, m, train, test, feats_all, True, use_gpu, weight_exponent, model_overrides)] = f"{m}_log"
                if extra_model_variants:
                    for variant in extra_model_variants:
                        merged_overrides = (model_overrides or {}) | variant.get("overrides", {})
                        futures[ex.submit(tree_predict, m, train, test, feats_all, False, use_gpu, weight_exponent, merged_overrides)] = f"{m}_{variant['suffix']}"
                        if use_log_target:
                            futures[ex.submit(tree_predict, m, train, test, feats_all, True, use_gpu, weight_exponent, merged_overrides)] = f"{m}_log_{variant['suffix']}"
            for fut in as_completed(futures):
                name = futures[fut]
                pred_df = fut.result()
                if not pred_df.empty:
                    preds_by_model[name].append(pred_df)

    results = []
    for name, frames in preds_by_model.items():
        if not frames:
            continue
        pred_df = pd.concat(frames, ignore_index=True).dropna(subset=["y_true", "y_pred"])
        if pred_df.empty:
            continue
        metrics = evaluate(pred_df["y_true"].values, pred_df["y_pred"].values)
        results.append(ModelResult(model_name=name, metrics=metrics, prediction_df=pred_df))
    return results
```

- [ ] **Step 3: Simplify `scripts/run_parallel_models_spend_t1.py` to import from core.py**

Remove the extracted functions and replace with imports. The script keeps:
- `main()` entry point and argument parsing (unchanged)
- `_baseline_residual_predict` and `_two_stage_regime_predict` (not used in current fast path anyway)

Replace the top of the file (lines 1-160, the extracted functions) and the ModelResult/BASE_FEATURE_COLS definitions with imports:

```python
# scripts/run_parallel_models_spend_t1.py (changes to top of file, lines 1-38)

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from app.experiments.core import (
    ModelResult,
    BASE_FEATURE_COLS,
    CALENDAR_PACING_FEATURE_COLS,
    feature_cols,
    safe_model_filename,
    evaluate,
    tree_predict,
    ewma_spend,
    run as _run,
)
from app.prediction_artifacts import validate_merged_daily_csv
from app.unified_daily import build_unified_daily

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None
```

Remove these now-redundant definitions from the script:
- `ModelResult` dataclass (line 34-37)
- `BASE_FEATURE_COLS` (lines 41-110)
- `CALENDAR_PACING_FEATURE_COLS` (lines 112-158)
- `_feature_cols` (lines 161-164)
- `_build_model` (lines 167-218)
- `_evaluate` (lines 221-226)
- `_safe_model_filename` (lines 229-230)
- `_evaluate_with_wmape` (lines 238-240)
- `_extract_app_id` (line 233-235) — dead code
- `_ewma_spend` (lines 393-406)
- `_tree_predict` (lines 409-447)
- `_apply_holiday_transition_calibration` (lines 450-491)
- `_baseline_residual_predict` (lines 494-528) — dead code, skip_baseline_two_stage=True always
- `_two_stage_regime_predict` (lines 531-607) — dead code, skip_baseline_two_stage=True always
- `_run` (lines 609-705)

Update all internal calls: `_evaluate` → `evaluate`, `_safe_model_filename` → `safe_model_filename`, `_feature_cols` → `feature_cols`, `_run` remains `_run` (via the import alias).

- [ ] **Step 4: Verify the refactor produces identical output**

```bash
cd /home/lsh/ad_ml && python scripts/run_parallel_models_spend_t1.py \
  --input daily_merged.csv \
  --output-dir outputs/model_parallel_spend_t1_refactor_test \
  --fast \
  --weight-exponent 0.35 \
  --eval-recent-days 5
```

Expected: Script completes without errors. Compare `metrics_summary.csv` with the baseline (spend v12) — the XGBoost_log MAPE should be in the same range (~62%) since no logic changed.

- [ ] **Step 5: Clean up verification output**

```bash
rm -rf outputs/model_parallel_spend_t1_refactor_test
```

- [ ] **Step 6: Commit**

```bash
git add app/experiments/__init__.py app/experiments/core.py scripts/run_parallel_models_spend_t1.py
git commit -m "refactor: extract shared training core into app.experiments.core"
```

---

### Task 2: Create experiment directory and config

**Files:**
- Create: `scripts/experiments/__init__.py`

- [ ] **Step 1: Create experiment directory**

```bash
mkdir -p scripts/experiments
```

```python
# scripts/experiments/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add scripts/experiments/__init__.py
git commit -m "chore: create experiments directory"
```

---

### Task 3: Write exp_01_quantile.py

**Files:**
- Create: `scripts/experiments/exp_01_quantile.py`

This experiment trains XGBoost models at 3 quantile levels (P10, P50, P90) on top of the standard MSE model. It compares coverage, interval width, and pinball loss against a no-quantile baseline.

- [ ] **Step 1: Write the experiment script**

```python
# scripts/experiments/exp_01_quantile.py
"""Experiment 01: Quantile Regression for Spend T+1 Prediction.

Trains 3 XGBoost quantile variants (alpha=0.1/0.5/0.9) alongside the standard
MSE XGBoost model. Evaluates calibration quality via coverage, pinball loss,
and normalized interval width — broken down by spend bucket.

Usage:
    python scripts/experiments/exp_01_quantile.py --input daily_merged.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.experiments.core import (
    ModelResult,
    evaluate,
    evaluate_pinball,
    compute_coverage,
    compute_interval_width,
    run,
)
from app.unified_daily import build_unified_daily
from app.prediction_artifacts import validate_merged_daily_csv

EXPERIMENT_NAME = "exp_01_quantile"
QUANTILE_ALPHAS = [0.1, 0.5, 0.9]


def _make_quantile_variants(alphas: List[float]) -> List[Dict]:
    """Build extra_model_variants entries for quantile regression."""
    variants = []
    for a in alphas:
        label = f"q{int(a * 100)}"
        variants.append({
            "suffix": label,
            "overrides": {
                "objective": "reg:quantileerror",
                "quantile_alpha": a,
            },
        })
    return variants


def _assign_spend_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each row to a spend bucket based on app-level mean spend."""
    app_avg = df.groupby("应用ID")["消耗金额"].mean().rename("app_avg_spend")
    df = df.join(app_avg, on="应用ID")
    df["spend_bucket"] = pd.qcut(df["app_avg_spend"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    return df


def _compute_quantile_metrics(results: List[ModelResult]) -> Dict:
    """Compute quantile-specific metrics from P10/P50/P90 predictions.

    Groups prediction DataFrames by date+app and computes coverage/width/pinball.
    """
    pred_dfs = {}
    for r in results:
        pred_dfs[r.model_name] = r.prediction_df.set_index(["日期", "应用ID"])

    # Find the P10/P50/P90 variants (prefer non-log)
    p10_key = None
    p50_key = None
    p90_key = None
    for key in pred_dfs:
        if "_q10" in key and "_log" not in key:
            p10_key = key
        elif "_q50" in key and "_log" not in key:
            p50_key = key
        elif "_q90" in key and "_log" not in key:
            p90_key = key

    if not all([p10_key, p50_key, p90_key]):
        return {"error": "Missing quantile predictions", "available_keys": list(pred_dfs.keys())}

    p10 = pred_dfs[p10_key]["y_pred"]
    p50 = pred_dfs[p50_key]["y_pred"]
    p90 = pred_dfs[p90_key]["y_pred"]
    y_true = pred_dfs[p10_key]["y_true"]

    coverage = compute_coverage(y_true.values, p10.values, p90.values)
    interval_width_median = float(np.median(compute_interval_width(p10.values, p90.values, p50.values)))

    pinball_scores = {}
    for a, key in [(0.1, p10_key), (0.5, p50_key), (0.9, p90_key)]:
        pinball_scores[f"alpha_{a}"] = evaluate_pinball(
            y_true.values, pred_dfs[key]["y_pred"].values, a
        )

    # MAPE on P50 (which is the "median" prediction — closest to point forecast)
    p50_metrics = evaluate(y_true.values, p50.values)

    return {
        "coverage_p10_p90": round(coverage, 4),
        "ideal_coverage": 0.80,
        "coverage_error_pp": round((coverage - 0.80) * 100, 2),
        "interval_width_median": round(interval_width_median, 4),
        "pinball_loss": {k: round(v, 6) for k, v in pinball_scores.items()},
        "p50_mape_pct": round(p50_metrics["mape_pct"], 2),
    }


def _bucket_metrics(results: List[ModelResult]) -> Dict:
    """Compute coverage/width within each spend bucket."""
    pred_dfs = {}
    for r in results:
        pred_dfs[r.model_name] = r.prediction_df

    merged = None
    for key, df in pred_dfs.items():
        df = df.copy()
        if merged is None:
            merged = df.rename(columns={"y_pred": f"pred_{key}", "y_true": "y_true"})
            merged = merged[["日期", "应用ID", "y_true", f"pred_{key}"]]
        else:
            sub = df[["日期", "应用ID", "y_pred"]].rename(columns={"y_pred": f"pred_{key}"})
            merged = merged.merge(sub, on=["日期", "应用ID"], how="inner")

    merged = _assign_spend_bucket(merged)

    p10_key = None
    p50_key = None
    p90_key = None
    for col in merged.columns:
        if col.startswith("pred_") and "_q10" in col and "_log" not in col:
            p10_key = col
        elif col.startswith("pred_") and "_q50" in col and "_log" not in col:
            p50_key = col
        elif col.startswith("pred_") and "_q90" in col and "_log" not in col:
            p90_key = col

    if not all([p10_key, p50_key, p90_key]):
        return {}

    bucket_stats = {}
    for bucket in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        b = merged[merged["spend_bucket"] == bucket]
        if b.empty:
            continue
        y = b["y_true"].values
        lo = b[p10_key].values
        md = b[p50_key].values
        hi = b[p90_key].values
        bucket_stats[bucket] = {
            "n": len(b),
            "coverage": round(compute_coverage(y, lo, hi), 4),
            "interval_width_median": round(float(np.median(compute_interval_width(lo, hi, md))), 4),
            "p50_mape_pct": round(evaluate(y, md)["mape_pct"], 2),
        }

    return bucket_stats


def main():
    parser = argparse.ArgumentParser(description="Exp 01: Quantile regression for T+1 spend")
    parser.add_argument("--input", default="daily_merged.csv")
    parser.add_argument("--eval-recent-days", type=int, default=20)
    parser.add_argument("--weight-exponent", type=float, default=0.35)
    args = parser.parse_args()

    validate_merged_daily_csv(Path(args.input))

    out_dir = Path("outputs/experiments") / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df, clean_meta = build_unified_daily(args.input, min_spend_train=2.0)

    quantile_variants = _make_quantile_variants(QUANTILE_ALPHAS)

    print("=== Training baseline + quantile variants ===")
    results = run(
        df,
        min_train_days=20,
        alpha=0.4,
        use_log_target=True,
        feature_set="base",
        eval_recent_days=args.eval_recent_days,
        tree_models=["XGBoost"],
        skip_baseline_two_stage=True,
        use_gpu=False,
        weight_exponent=args.weight_exponent,
        extra_model_variants=quantile_variants,
    )

    if not results:
        raise ValueError("No results produced. Check data and parameters.")

    # Save predictions
    for r in results:
        filename = f"predictions_{r.model_name.replace('/', '_')}.csv"
        r.prediction_df.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")

    # Baseline metrics (standard MSE XGBoost)
    baseline_rows = []
    for r in results:
        if "_q" not in r.model_name:  # standard models only
            baseline_rows.append({
                "model": r.model_name,
                **r.metrics,
                "samples": int(len(r.prediction_df)),
            })
    baseline_df = pd.DataFrame(baseline_rows).sort_values("mape")
    baseline_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    # Quantile-specific metrics
    quantile_metrics = _compute_quantile_metrics(results)
    bucket_stats = _bucket_metrics(results)

    # Verdict
    coverage_err = abs(quantile_metrics.get("coverage_error_pp", 999))
    if coverage_err <= 5:
        verdict = "promising"
    elif coverage_err <= 15:
        verdict = "inconclusive"
    else:
        verdict = "regress"

    report = {
        "experiment": EXPERIMENT_NAME,
        "baseline": "spend_v12_unified (MSE XGBoost)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_recent_days": args.eval_recent_days,
            "weight_exponent": args.weight_exponent,
            "quantile_alphas": QUANTILE_ALPHAS,
        },
        "baseline_best_mape_pct": float(baseline_df.iloc[0]["mape_pct"]) if len(baseline_df) > 0 else None,
        "quantile_metrics": quantile_metrics,
        "by_spend_bucket": bucket_stats,
        "verdict": verdict,
    }

    report_path = out_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print(f"  Coverage (P10-P90): {quantile_metrics.get('coverage_p10_p90', 'N/A')}")
    print(f"  Interval width (median): {quantile_metrics.get('interval_width_median', 'N/A')}")
    if bucket_stats:
        print(f"  Q4_high coverage: {bucket_stats.get('Q4_high', {}).get('coverage', 'N/A')}")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

```bash
cd /home/lsh/ad_ml && python scripts/experiments/exp_01_quantile.py \
  --input daily_merged.csv \
  --eval-recent-days 20 \
  --weight-exponent 0.35
```

Expected: Script completes and writes results to `outputs/experiments/exp_01_quantile/`.

- [ ] **Step 3: Inspect the results**

```bash
cat outputs/experiments/exp_01_quantile/experiment_report.json
cat outputs/experiments/exp_01_quantile/metrics_summary.csv
```

Expected: `experiment_report.json` shows coverage, pinball loss, interval width, by-bucket stats, and verdict.

- [ ] **Step 4: Commit**

```bash
git add scripts/experiments/exp_01_quantile.py
git commit -m "feat: add exp_01 quantile regression experiment (P10/P50/P90)"
```

---

### Task 4: Quick rollback test

Verify the refactored `run_parallel_models_spend_t1.py` still works correctly standalone (no experiment imports needed for this to pass).

- [ ] **Step 1: Run original script after refactor**

```bash
cd /home/lsh/ad_ml && python scripts/run_parallel_models_spend_t1.py \
  --input daily_merged.csv \
  --output-dir outputs/model_parallel_spend_t1_rollback_test \
  --fast \
  --weight-exponent 0.35 \
  --eval-recent-days 5
```

Expected: Completes without errors, `metrics_summary.csv` has XGBoost_log MAPE ~62%.

- [ ] **Step 2: Clean up**

```bash
rm -rf outputs/model_parallel_spend_t1_rollback_test
```

- [ ] **Step 3: Report**

No commit needed — this is verification only. Confirm:
- Original script: works, unchanged output
- Experiment script: isolated in `outputs/experiments/exp_01_quantile/`
- Rollback: delete `outputs/experiments/exp_01_quantile/` + `scripts/experiments/exp_01_quantile.py`
