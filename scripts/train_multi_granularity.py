"""统一多粒度训练脚本：支持 应用/广告主/广告组 三种粒度"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.training_config import APP_CONFIG, ACCOUNT_CONFIG, ADGROUP_CONFIG

CONFIGS = {"app": APP_CONFIG, "account": ACCOUNT_CONFIG, "adgroup": ADGROUP_CONFIG}


def load_aggregate(cfg):
    """加载数据并聚合到 entity-日期级"""
    pq = Path(cfg.data_path).with_suffix('.parquet')
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(cfg.data_path)
    df["日期"] = pd.to_datetime(df["日期"])
    cols = [cfg.entity_col, "日期", "消耗金额", "首日广告收入"]
    df = df[[c for c in cols if c in df.columns]]
    ad = df.groupby([cfg.entity_col, "日期"], as_index=False).agg(
        spend=("消耗金额", "sum"), d1_rev=("首日广告收入", "sum"))
    return ad[ad["spend"] >= cfg.min_spend_train].copy()


def build_features(ad, cfg):
    """构建通用特征"""
    ec = cfg.entity_col
    ad["dow"] = ad["日期"].dt.weekday
    ad["dom"] = ad["日期"].dt.day
    ad["month"] = ad["日期"].dt.month
    ad["roi"] = ad["d1_rev"] / np.maximum(ad["spend"], 1e-8)
    ad = ad.sort_values([ec, "日期"])
    for lag in [1, 2, 3, 7]:
        ad[f"spend_lag{lag}"] = ad.groupby(ec)["spend"].shift(lag)
        ad[f"roi_lag{lag}"] = ad.groupby(ec)["roi"].shift(lag)
    for w in [3, 7]:
        ad[f"spend_roll{w}"] = ad.groupby(ec)["spend"].transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        ad[f"roi_roll{w}"] = ad.groupby(ec)["roi"].transform(
            lambda x: x.rolling(w, min_periods=1).mean())
    return ad


def train_and_save(ad, cfg):
    """训练并保存模型"""
    feats = [c for c in ad.columns if c not in
             [cfg.entity_col, "日期", "d1_rev", "target_spend", "target_roi"]]
    for h in range(1, cfg.output_chunk_length + 1):
        ad[f"target_spend_h{h}"] = ad.groupby(cfg.entity_col)["spend"].shift(-h)
    ad = ad.dropna(subset=[f"target_spend_h{h}" for h in range(1, cfg.output_chunk_length + 1)] + feats)

    X = ad[feats].values
    Y = ad[[f"target_spend_h{h}" for h in range(1, cfg.output_chunk_length + 1)]].values
    mask = np.isfinite(Y).all(axis=1)
    X, Y = X[mask], Y[mask]

    dates = ad["日期"].values[mask]
    cut = pd.Timestamp("2026-05-01")
    train_idx = dates < cut
    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_te, Y_te = X[~train_idx], Y[~train_idx]
    print(f"  训练: {len(X_tr)}, 测试: {len(X_te)}, 特征: {len(feats)}")

    out = Path(cfg.full_output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for h in range(cfg.output_chunk_length):
        m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=6)
        m.fit(X_tr, Y_tr[:, h])
        m.save_model(str(out / f"spend_h{h+1}.json"))
        p = m.predict(X_te)
        wm = np.sum(np.abs(Y_te[:, h] - p)) / max(np.sum(np.abs(Y_te[:, h])), 1) * 100
        corr = np.corrcoef(Y_te[:, h], p)[0, 1] if len(p) > 2 else 0
        print(f"  T+{h+1}: WMAPE={wm:.1f}% Corr={corr:.3f}")
    json.dump({"features": feats, "entity": cfg.entity_col, "samples": len(X)},
              open(out / "meta.json", "w"), ensure_ascii=False)
    return len(X), feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--granularity", choices=["app","account","adgroup"], default="account")
    parser.add_argument("--horizons", type=int, default=1)
    args = parser.parse_args()

    cfg = CONFIGS[args.granularity]
    cfg.output_chunk_length = args.horizons
    print(f"=== {cfg.entity_col} 粒度训练, {args.horizons}步预测 ===")

    t0 = time.time()
    ad = load_aggregate(cfg)
    print(f"数据: {len(ad)}行, {ad[cfg.entity_col].nunique()}实体")
    ad = build_features(ad, cfg)
    n, feats = train_and_save(ad, cfg)
    print(f"✅ 完成: {n}样本, {len(feats)}特征, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
