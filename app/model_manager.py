"""模型管理器：懒加载 + TTL缓存 + 多版本"""
import time, json, threading
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class ModelManager:
    """按需加载模型，过期自动卸载"""

    def __init__(self, ttl_seconds: int = 1800):
        self._ttl = ttl_seconds
        self._models: Dict[str, List] = {}   # {model_key: [model_h1, model_h2, ...]}
        self._features: Dict[str, List[str]] = {}  # {model_key: [feat_cols]}
        self._last_access: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _model_key(self, granularity: str, version: str = "v1") -> str:
        return f"{granularity}_{version}"

    def load_if_needed(self, granularity: str, version: str = "v1",
                       model_dir: Optional[Path] = None) -> bool:
        """按需加载模型，已加载则刷新TTL"""
        import xgboost as xgb
        key = self._model_key(granularity, version)
        with self._lock:
            if key in self._models:
                self._last_access[key] = time.time()
                return True
            if model_dir is None:
                # Auto-detect model dir
                base = Path("outputs")
                candidates = [
                    base / f"model_{granularity}",
                    base / "multi_horizon_models",
                ]
                model_dir = next((d for d in candidates if d.exists()), None)
            if model_dir is None or not model_dir.exists():
                return False
            try:
                meta = json.loads((model_dir / "meta.json").read_text())
            except Exception:
                meta = json.loads((model_dir / "features.json").read_text()) if (model_dir / "features.json").exists() else {}
            feats = meta.get("features", [])
            models = []
            for h in range(1, 8):
                mf = model_dir / f"spend_h{h}.json"
                if not mf.exists():
                    break
                m = xgb.XGBRegressor()
                m.load_model(str(mf))
                models.append(m)
            if not models:
                return False
            self._models[key] = models
            self._features[key] = feats
            self._last_access[key] = time.time()
            return True

    def predict(self, granularity: str, features: np.ndarray,
                version: str = "v1", model_dir: Optional[Path] = None) -> Optional[List[float]]:
        """预测 T+1..T+N，自动加载模型"""
        if not self.load_if_needed(granularity, version, model_dir):
            return None
        key = self._model_key(granularity, version)
        with self._lock:
            self._last_access[key] = time.time()
            return [float(m.predict(features)[0]) for m in self._models[key]]

    def predict_horizon(self, granularity: str, features: np.ndarray, horizon: int,
                        version: str = "v1", model_dir: Optional[Path] = None) -> Optional[float]:
        """预测单个horizon"""
        preds = self.predict(granularity, features, version, model_dir)
        if preds and horizon <= len(preds):
            return preds[horizon - 1]
        return None

    def unload_stale(self) -> int:
        """卸载超过TTL未访问的模型，返回卸载数量"""
        now = time.time()
        with self._lock:
            stale = [k for k, t in self._last_access.items() if now - t > self._ttl]
            for k in stale:
                del self._models[k]
                del self._features[k]
                del self._last_access[k]
            return len(stale)

    def unload_all(self):
        with self._lock:
            self._models.clear()
            self._features.clear()
            self._last_access.clear()

    @property
    def loaded_models(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())

    @property
    def memory_mb(self) -> float:
        """估算内存占用(MB)"""
        total = 0
        with self._lock:
            for models in self._models.values():
                for m in models:
                    # XGBoost模型大小估算
                    try:
                        total += m.get_booster().save_raw().__sizeof__() / 1024 / 1024
                    except Exception:
                        total += 1.0  # 每个模型约1MB
        return total


# 全局单例
model_mgr = ModelManager(ttl_seconds=1800)
