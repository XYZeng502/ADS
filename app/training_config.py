"""统一训练配置：支持应用/广告主/广告组等多粒度训练"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """一次训练运行的全部配置"""

    # 数据
    data_path: str = "daily_merged.parquet"
    entity_col: str = "应用ID"  # 粒度：应用ID / 广告主ID / 广告组ID
    min_spend_train: float = 5.0
    min_entity_days: int = 30

    # 模型
    tree_models: List[str] = field(default_factory=lambda: ["XGBoost"])
    use_gpu: bool = False
    weight_exponent: float = 0.35

    # 训练
    min_train_days: int = 20
    eval_recent_days: int = 20
    output_chunk_length: int = 1  # 1=T+1 only, 7=T+1..T+7 multi-horizon

    # 输出
    output_dir: str = "outputs/model_run"

    @property
    def entity_name(self) -> str:
        names = {"应用ID": "app", "广告主ID": "account", "广告组ID": "adgroup"}
        return names.get(self.entity_col, "entity")

    @property
    def full_output_dir(self) -> str:
        return f"{self.output_dir}_{self.entity_name}_{self.output_chunk_length}h"


# 预定义配置
APP_CONFIG = TrainingConfig(
    entity_col="应用ID",
    min_spend_train=5.0,
    output_dir="outputs/model_app",
)

ACCOUNT_CONFIG = TrainingConfig(
    entity_col="广告主ID",
    min_spend_train=10.0,
    min_entity_days=20,
    output_dir="outputs/model_account",
)

ADGROUP_CONFIG = TrainingConfig(
    entity_col="广告组ID",
    min_spend_train=20.0,
    min_entity_days=15,
    output_dir="outputs/model_adgroup",
)
