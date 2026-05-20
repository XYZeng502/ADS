# CQR Production Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate CQR per-bucket calibrated prediction intervals into the T+1 spend training pipeline, web dashboard, and budget recommendations.

**Architecture:** CQR runs as a post-processing step after the standard walk-forward training. It takes 3 quantile variant predictions (XGBoost_q5/q50/q95), assigns static spend buckets, calibrates per-bucket with MAPIE CQR, and outputs a single CSV with interval columns. Web dashboard reads this CSV and renders interval bands on ECharts. Budget recommendations derive three tiers from P05/P50/P95 spend.

**Tech Stack:** Python 3, XGBoost (`reg:quantileerror`), MAPIE (`ConformalizedQuantileRegressor`), pandas, ECharts (JavaScript), FastAPI

---

### Task 1: Add `build_cqr_prediction()` to `app/experiments/core.py`

**Files:**
- Modify: `app/experiments/core.py`

**Goal:** Extract the per-bucket CQR calibration logic from `scripts/experiments/exp_01c_cqr.py` into a reusable function that takes three quantile prediction DataFrames and returns calibrated interval predictions.

- [ ] **Step 1: Read existing quantile prediction format from experiments**

```bash
PYTHONPATH=/home/lsh/ad_ml python -c "
import pandas as pd
# Check what format quantile variant predictions have
df = pd.read_csv('outputs/experiments/exp_01_quantile/predictions_XGBoost_q5.csv')
print('Columns:', list(df.columns))
print('Models:', df['model'].unique())
print('Head:', df.head(2).to_dict())
"
```

- [ ] **Step 2: Add `build_cqr_prediction()` to core.py**

Insert after the existing `assign_spend_bucket` function. Read the existing `app/experiments/core.py` first to find the right insertion point.

```python
# Add at end of app/experiments/core.py, before the final line:

def build_cqr_prediction(pred_q05: pd.DataFrame,
                         pred_q50: pd.DataFrame,
                         pred_q95: pd.DataFrame,
                         weight_exponent: float = 0.35,
                         q4_confidence: float = 0.85,
                         random_state: int = 42) -> pd.DataFrame:
    """Post-hoc CQR calibration on quantile variant predictions.

    Takes three quantile XGBoost prediction DataFrames (from
    extra_model_variants with alpha=0.05/0.5/0.95), applies per-bucket
    CQR calibration, and returns a single DataFrame with calibrated
    point predictions and 80% prediction intervals.

    Args:
        pred_q05: DataFrame(日期, 应用ID, y_true, y_pred, model) — alpha=0.05
        pred_q50: DataFrame(日期, 应用ID, y_true, y_pred, model) — alpha=0.5
        pred_q95: DataFrame(日期, 应用ID, y_true, y_pred, model) — alpha=0.95
        q4_confidence: Target coverage for Q4 bucket (others use 0.80)

    Returns:
        DataFrame(日期, 应用ID, y_true, y_pred, y_lower, y_upper, model="XGBoost_CQR")
    """
    try:
        from mapie.regression import ConformalizedQuantileRegressor
    except ImportError:
        raise ImportError("MAPIE is required for CQR. Install: pip install mapie")

    df_q05 = pred_q05.copy()
    df_q50 = pred_q50.copy()
    df_q95 = pred_q95.copy()

    # Assign static spend buckets (based on overall y_true mean per app)
    all_y_true = pd.concat([df_q05[["日期", "应用ID", "y_true"]],
                            df_q50[["日期", "应用ID", "y_true"]],
                            df_q95[["日期", "应用ID", "y_true"]]])
    app_avg = all_y_true.groupby("应用ID")["y_true"].mean()
    bucket_map = pd.qcut(app_avg, q=4, labels=SPEND_BUCKET_LABELS)

    for df_part in [df_q05, df_q50, df_q95]:
        df_part["_bucket"] = df_part["应用ID"].map(bucket_map)

    # Build per-bucket confidence levels
    bucket_conf = {b: (q4_confidence if b == "Q4_high" else 0.80)
                   for b in SPEND_BUCKET_LABELS}

    results = []
    # Process one unique date at a time (walk-forward context already built in)
    for dt in sorted(df_q05["日期"].unique()):
        mask05 = df_q05["日期"] == dt
        mask50 = df_q50["日期"] == dt
        mask95 = df_q95["日期"] == dt

        sub05 = df_q05[mask05].copy()
        sub50 = df_q50[mask50].copy()
        sub95 = df_q95[mask95].copy()

        if sub05.empty or sub50.empty or sub95.empty:
            continue

        # Build quantile prediction matrix: (n, 3) with cols [P05, P50, P95]
        sub05_sorted = sub05.set_index("应用ID").sort_index()
        sub50_sorted = sub50.set_index("应用ID").sort_index()
        sub95_sorted = sub95.set_index("应用ID").sort_index()

        common_apps = sub05_sorted.index.intersection(sub50_sorted.index)
        common_apps = common_apps.intersection(sub95_sorted.index)

        if len(common_apps) < 5:
            continue

        y_lower_all = np.full(len(common_apps), np.nan)
        y_upper_all = np.full(len(common_apps), np.nan)
        y_pred_all = sub50_sorted.loc[common_apps, "y_pred"].values
        y_true_all = sub05_sorted.loc[common_apps, "y_true"].values
        buckets_all = sub05_sorted.loc[common_apps, "_bucket"].values

        q_preds = np.column_stack([
            sub05_sorted.loc[common_apps, "y_pred"].values,
            sub50_sorted.loc[common_apps, "y_pred"].values,
            sub95_sorted.loc[common_apps, "y_pred"].values,
        ])

        # Per-bucket CQR calibration
        for bucket in SPEND_BUCKET_LABELS:
            bm = buckets_all == bucket
            if bm.sum() < 10:
                continue

            q_bucket = q_preds[bm]
            y_bucket = y_true_all[bm]

            # Split: calibration set (30%) within this date-bucket
            n = len(q_bucket)
            n_calib = max(int(n * 0.3), 5)
            np.random.seed(random_state)
            perm = np.random.permutation(n)
            x_calib = q_bucket[perm[:n_calib]]
            y_calib = y_bucket[perm[:n_calib]]
            x_test = q_bucket[perm[n_calib:]]

            # Sort columns for CQR (P05, P50, P95 order) and ensure non-crossing
            x_calib_sorted = np.sort(x_calib, axis=1)
            x_test_sorted = np.sort(x_test, axis=1)

            try:
                cqr = ConformalizedQuantileRegressor(
                    estimator=None,
                    confidence_level=bucket_conf[bucket],
                )
                cqr.fit(x_calib_sorted, y_calib)
                _, y_pis = cqr.predict(x_test_sorted)
                y_pis = y_pis.squeeze(-1)

                test_indices = np.where(bm)[0][perm[n_calib:]]
                y_lower_all[test_indices] = np.min(y_pis, axis=1)
                y_upper_all[test_indices] = np.max(y_pis, axis=1)
            except Exception:
                # Fall back to raw quantile predictions if CQR fails
                test_indices = np.where(bm)[0][perm[n_calib:]]
                raw_sorted = np.sort(q_bucket[perm[n_calib:]], axis=1)
                y_lower_all[test_indices] = raw_sorted[:, 0]
                y_upper_all[test_indices] = raw_sorted[:, 2]

        # Only keep rows where calibration succeeded
        valid = ~np.isnan(y_lower_all) & ~np.isnan(y_upper_all)
        if valid.sum() == 0:
            continue

        results.append(pd.DataFrame({
            "日期": dt,
            "应用ID": common_apps[valid],
            "y_true": y_true_all[valid],
            "y_pred": np.clip(y_pred_all[valid], 0.0, None),
            "y_lower": np.clip(y_lower_all[valid], 0.0, None),
            "y_upper": np.clip(y_upper_all[valid], 0.0, None),
            "model": "XGBoost_CQR",
        }))

    if not results:
        raise ValueError("CQR calibration produced no results. Check input predictions.")

    return pd.concat(results, ignore_index=True)
```

- [ ] **Step 3: Verify syntax and test on existing experiment data**

```bash
PYTHONPATH=/home/lsh/ad_ml python -c "
from app.experiments.core import build_cqr_prediction
print('Function imported OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add app/experiments/core.py
git commit -m "feat: add build_cqr_prediction() — per-bucket CQR post-hoc calibration"
```

---

### Task 2: Integrate CQR into run_parallel_models_spend_t1.py

**Files:**
- Modify: `scripts/run_parallel_models_spend_t1.py`

**Goal:** Add `--enable-cqr` and `--q4-confidence` flags. When enabled, pass quantile extra_model_variants to `_run()`, then call `build_cqr_prediction()` on the resulting predictions.

- [ ] **Step 1: Read the current script to understand the main() flow**

```bash
wc -l scripts/run_parallel_models_spend_t1.py
```

- [ ] **Step 2: Add CLI arguments**

After the existing `--weight-exponent` argument (near line 290), insert:

```python
    parser.add_argument(
        "--enable-cqr",
        action="store_true",
        help="Enable CQR per-bucket calibration for prediction intervals (requires mapie).",
    )
    parser.add_argument(
        "--q4-confidence",
        type=float,
        default=0.85,
        help="Target coverage for Q4 high-spend bucket (default 0.85).",
    )
```

- [ ] **Step 3: Add CQR extra_model_variants construction**

After the tree_override logic (near the `_run()` call), insert:

```python
    extra_variants = None
    if args.enable_cqr:
        extra_variants = [
            {"suffix": "q5",  "overrides": {"objective": "reg:quantileerror", "quantile_alpha": 0.05}},
            {"suffix": "q50", "overrides": {"objective": "reg:quantileerror", "quantile_alpha": 0.5}},
            {"suffix": "q95", "overrides": {"objective": "reg:quantileerror", "quantile_alpha": 0.95}},
        ]
```

- [ ] **Step 4: Pass extra_model_variants to _run()**

Find the `_run()` call in main() and add the parameter:

```python
    results = _run(
        df,
        ...
        weight_exponent=args.weight_exponent,
        extra_model_variants=extra_variants,   # add this line
    )
```

- [ ] **Step 5: Add CQR post-processing after the main results loop**

After the existing prediction-saving loop (after `r.prediction_df.to_csv(...)`), add:

```python
    # CQR post-hoc calibration
    cqr_report = None
    if args.enable_cqr:
        from app.experiments.core import build_cqr_prediction, compute_coverage, evaluate
        preds_by_name = {r.model_name: r.prediction_df for r in results}
        q5_key = "XGBoost_q5"
        q50_key = "XGBoost_q50"
        q95_key = "XGBoost_q95"
        if q5_key in preds_by_name and q50_key in preds_by_name and q95_key in preds_by_name:
            cqr_df = build_cqr_prediction(
                pred_q05=preds_by_name[q5_key],
                pred_q50=preds_by_name[q50_key],
                pred_q95=preds_by_name[q95_key],
                weight_exponent=args.weight_exponent,
                q4_confidence=args.q4_confidence,
            )
            cqr_path = out_dir / "predictions_XGBoost_CQR.csv"
            cqr_df.to_csv(cqr_path, index=False, encoding="utf-8-sig")

            y_true = cqr_df["y_true"].values
            y_lower = cqr_df["y_lower"].values
            y_upper = cqr_df["y_upper"].values
            y_pred = cqr_df["y_pred"].values
            cqr_metrics = evaluate(y_true, y_pred)
            cqr_coverage = float(compute_coverage(y_true, y_lower, y_upper))

            cqr_report = {
                "coverage": round(cqr_coverage, 4),
                "mape_pct": round(cqr_metrics["mape_pct"], 2),
                "q4_confidence": args.q4_confidence,
            }
            print(f"CQR: coverage={cqr_coverage:.4f} MAPE={cqr_metrics['mape_pct']:.2f}%")
        else:
            print("CQR: quantile variants not found in results, skipping calibration")
```

- [ ] **Step 6: Add cqr_report to the report.json output**

In the report dict, add:

```python
    "cqr": cqr_report,
```

- [ ] **Step 7: Run a quick smoke test**

```bash
PYTHONPATH=/home/lsh/ad_ml python scripts/run_parallel_models_spend_t1.py \
  --input daily_merged.csv \
  --output-dir outputs/model_parallel_spend_t1_cqr_test \
  --tree-models XGBoost \
  --use-log-target \
  --enable-cqr \
  --eval-recent-days 5 \
  --weight-exponent 0.35 2>&1 | tail -20
```

Expected: Script completes, outputs `predictions_XGBoost_CQR.csv` with y_lower/y_upper columns, CQR coverage ~0.75-0.85.

- [ ] **Step 8: Clean up and commit**

```bash
rm -rf outputs/model_parallel_spend_t1_cqr_test
git add scripts/run_parallel_models_spend_t1.py
git commit -m "feat: add --enable-cqr flag for prediction intervals in spend T+1"
```

---

### Task 3: Update daily_retrain.py

**Files:**
- Modify: `scripts/daily_retrain.py`

**Goal:** Add `--enable-cqr` to the spend training step.

- [ ] **Step 1: Read the current spend step definition**

```bash
grep -A 15 "spend_t1" scripts/daily_retrain.py
```

- [ ] **Step 2: Add --enable-cqr flag to the spend step cmd**

```python
    {
        "id": "spend_t1",
        "label": "Spend T+1 预测",
        "cmd": [
            sys.executable, "scripts/run_parallel_models_spend_t1.py",
            "--input", "daily_merged.csv",
            "--output-dir", "outputs/model_parallel_spend_t1",
            "--tree-models", "XGBoost",
            "--use-log-target",
            "--enable-cqr",
            "--eval-recent-days", "40",
        ],
    },
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "compile(open('scripts/daily_retrain.py').read(), 'daily_retrain.py', 'exec'); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/daily_retrain.py
git commit -m "feat: enable CQR intervals in daily spend retraining step"
```

---

### Task 4: Web dashboard interval display

**Files:**
- Modify: `app/web.py`

**Goal:** Load CQR predictions and render interval bands on the spend chart. Add interval columns to the prediction table.

- [ ] **Step 1: Add CQR data loading to `_load_model_dashboard_payload()`**

Find this function (~line 237) and after the existing spend prediction loading, add:

```python
    # CQR interval predictions
    cqr_path = spend_dir / "predictions_XGBoost_CQR.csv" if spend_dir else None
    cqr_predictions = None
    cqr_coverage = None
    if cqr_path and cqr_path.exists():
        cqr_raw = _read_csv_rows(cqr_path, 50000, from_end=True)
        # Build date-indexed interval data for chart
        cqr_by_date: Dict[str, Dict] = {}
        for row in cqr_raw:
            day = str(row.get("日期", ""))
            if day:
                cqr_by_date[day] = {
                    "y_pred": float(row.get("y_pred", 0)),
                    "y_lower": float(row.get("y_lower", 0)),
                    "y_upper": float(row.get("y_upper", 0)),
                }
        cqr_predictions = cqr_by_date
        # Read coverage from report
        report = _read_json(spend_dir / "report.json") if spend_dir else {}
        cqr_data = report.get("cqr", {})
        cqr_coverage = cqr_data.get("coverage")
```

- [ ] **Step 2: Add CQR data to the spend payload dict**

In the return dict under `"spend"`, add:

```python
    "cqr_predictions": cqr_predictions,
    "cqr_coverage": cqr_coverage,
```

- [ ] **Step 3: Add interval bands to the spend ECharts config**

Find the JavaScript section where spend chart series are defined. Add the interval band series before the y_pred line:

```javascript
    // CQR interval band (add before the y_pred series if cqr data available)
    if (spendData.cqr_predictions) {
        const cqrLower = dates.map(d => {
            const r = spendData.cqr_predictions[d];
            return r ? r.y_lower : null;
        });
        const cqrUpper = dates.map(d => {
            const r = spendData.cqr_predictions[d];
            return r ? r.y_upper : null;
        });
        
        series.push({
            name: '预测区间',
            type: 'line',
            data: cqrLower,
            lineStyle: {opacity: 0},
            stack: 'cqr-band',
            symbol: 'none',
            silent: true,
        });
        series.push({
            name: '区间上界',
            type: 'line',
            data: cqrUpper,
            lineStyle: {opacity: 0},
            areaStyle: {color: 'rgba(66,133,244,0.12)'},
            stack: 'cqr-band',
            symbol: 'none',
            silent: true,
        });
    }
```

- [ ] **Step 4: Update tooltip to show interval**

Find the tooltip formatter for the spend chart and update it to show interval when CQR data is available:

```javascript
    formatter: function(params) {
        const date = params[0].axisValue;
        let html = '<b>' + date + '</b><br/>';
        params.forEach(p => {
            html += p.marker + ' ' + p.seriesName + ': ' + p.value + '<br/>';
        });
        // Add interval info if CQR data exists
        if (spendData.cqr_predictions && spendData.cqr_predictions[date]) {
            const c = spendData.cqr_predictions[date];
            html += '区间: ' + c.y_lower.toFixed(0) + ' - ' + c.y_upper.toFixed(0) + '<br/>';
        }
        return html;
    }
```

- [ ] **Step 5: Commit**

```bash
git add app/web.py
git commit -m "feat: add CQR interval bands to spend prediction chart"
```

---

### Task 5: Three-tier budget recommendations

**Files:**
- Modify: `scripts/offline_backtest.py`
- Modify: `app/web.py`

**Goal:** Compute conservative/neutral/aggressive budget tiers from CQR spend intervals.

- [ ] **Step 1: Add tiered budget calculation to `run_app_level_last_day_prediction()`**

In `scripts/offline_backtest.py`, find the function and after existing budget calculation, add:

```python
    # Three-tier budget recommendations using CQR intervals
    tiered_budgets = []
    cqr_path = Path("outputs/model_parallel_spend_t1/predictions_XGBoost_CQR.csv")
    if cqr_path.exists():
        cqr_df = pd.read_csv(cqr_path, encoding="utf-8-sig")
        roi_path = resolve_roi_predictions_csv(
            _pick_existing([
                root / "model_parallel_roi_d1_v9_unified",
                root / "model_parallel_roi_d1_v8_001",
            ])
        )
        if roi_path and roi_path.exists():
            roi_df = pd.read_csv(roi_path, encoding="utf-8-sig")
            # Get latest day predictions per app
            latest_spend = cqr_df.sort_values("日期").groupby("应用ID").tail(1)
            latest_roi = roi_df.sort_values("日期").groupby("应用ID").tail(1)
            
            for _, spend_row in latest_spend.iterrows():
                app_id = spend_row["应用ID"]
                roi_row = latest_roi[latest_roi["应用ID"] == app_id]
                if roi_row.empty:
                    continue
                roi_pred = max(float(roi_row.iloc[0]["y_pred"]), 0.001)
                
                tiered_budgets.append({
                    "应用ID": str(app_id),
                    "conservative_spend": float(spend_row["y_lower"]),
                    "neutral_spend": float(spend_row["y_pred"]),
                    "aggressive_spend": float(spend_row["y_upper"]),
                    "roi_anchor": roi_pred,
                    "conservative_budget": float(spend_row["y_lower"]) * roi_pred,
                    "neutral_budget": float(spend_row["y_pred"]) * roi_pred,
                    "aggressive_budget": float(spend_row["y_upper"]) * roi_pred,
                })
    
    # Save tiered recommendations
    if tiered_budgets:
        tiered_path = output_dir / "tiered_budget_recommendations.csv"
        pd.DataFrame(tiered_budgets).to_csv(tiered_path, index=False, encoding="utf-8-sig")
```

- [ ] **Step 2: Add tiered budget display to web.py**

In the web dashboard, add a section that reads `tiered_budget_recommendations.csv` and displays a table with three columns (conservative/neutral/aggressive) per app. Add a toggle or selector to switch between tiers.

- [ ] **Step 3: Commit**

```bash
git add scripts/offline_backtest.py app/web.py
git commit -m "feat: three-tier budget recommendations from CQR spend intervals"
```

---

### Task 6: End-to-end verification

**Goal:** Run full pipeline with CQR enabled, verify all outputs are correct, and check web dashboard renders properly.

- [ ] **Step 1: Run the full spend training pipeline with CQR**

```bash
PYTHONPATH=/home/lsh/ad_ml python scripts/run_parallel_models_spend_t1.py \
  --input daily_merged.csv \
  --output-dir outputs/model_parallel_spend_t1_cqr_e2e \
  --tree-models XGBoost \
  --use-log-target \
  --enable-cqr \
  --q4-confidence 0.85 \
  --eval-recent-days 20 \
  --weight-exponent 0.35 2>&1 | grep -v "^INFO:root" | tail -30
```

- [ ] **Step 2: Verify CQR output file structure**

```bash
PYTHONPATH=/home/lsh/ad_ml python -c "
import pandas as pd
df = pd.read_csv('outputs/model_parallel_spend_t1_cqr_e2e/predictions_XGBoost_CQR.csv')
print('Columns:', list(df.columns))
print('Rows:', len(df))
print('Models:', df['model'].unique())
print('Null check:')
print(df[['y_lower','y_upper']].isnull().sum())
print()
print('Sample (last 5 rows):')
print(df.tail())
print()
# Verify intervals are valid
bad = (df['y_lower'] > df['y_upper']).sum()
print(f'Rows with lower > upper: {bad}')
in_interval = (df['y_true'] >= df['y_lower']) & (df['y_true'] <= df['y_upper'])
print(f'Coverage: {in_interval.mean():.4f}')
"
```

Expected: No nulls in y_lower/y_upper, lower <= upper always, coverage 0.75-0.87.

- [ ] **Step 3: Verify web dashboard loads**

Start the web server and check that the spend chart renders with interval bands:

```bash
# Start server in background
PYTHONPATH=/home/lsh/ad_ml bash scripts/run_web_8000.sh &
sleep 3
# Check the spend API endpoint returns CQR data
curl -s http://localhost:8000/web/predict | python -c "import sys,json; d=json.load(sys.stdin); s=d['spend']; print('Has CQR:', s.get('cqr_coverage') is not None)"
# Kill server
pkill -f "uvicorn app.main"
```

- [ ] **Step 4: Clean up test outputs**

```bash
rm -rf outputs/model_parallel_spend_t1_cqr_e2e
```

- [ ] **Step 5: Final commit if any fixes needed**

```bash
git status
# Commit any remaining changes
```
