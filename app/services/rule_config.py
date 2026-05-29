"""用户自定义推荐规则：JSON文件存储 + 条件匹配 + 动作覆盖

规则由广告优化师通过 Web 界面配置。每条规则定义一组条件，
命中后触发对应动作（放量/控量/调价等），结果直接体现在推荐 CSV 和表格中。
"""
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── 动作类型 ──

VALID_ACTIONS = [
    "SCALE",        # 放量：提升预算，扩大投放
    "GUARD",        # 控量：缩减预算，保守投放
    "STABLE",       # 维稳：维持当前节奏
    "PAUSE",        # 暂停：暂停该应用投放
    "BOOST_BUDGET", # 加预算：在建议预算基础上额外增加指定比例
    "CUT_BUDGET",   # 减预算：在建议预算基础上额外减少指定比例
    "ADJUST_BID",   # 调整出价：降低或提高出价
]

ACTION_LABELS = {
    "SCALE": "放量",
    "GUARD": "控量",
    "STABLE": "维稳",
    "PAUSE": "暂停投放",
    "BOOST_BUDGET": "加预算",
    "CUT_BUDGET": "减预算",
    "ADJUST_BID": "调整出价",
}


class RuleCondition(BaseModel):
    field: str
    op: Literal[">=", "<=", ">", "<", "==", "!="]
    value: float | str


class RecommendationRule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    enabled: bool = True
    app_match: str = "*"
    conditions: List[RuleCondition] = Field(default_factory=list)
    action: str = "SCALE"  # one of VALID_ACTIONS
    budget_adjust_ratio: float = 0.0  # for BOOST_BUDGET/CUT_BUDGET: e.g. 0.2 = +20%
    reason: str = ""
    priority: int = 100


class RuleConfigStore:
    def __init__(self, path: Path | str = "outputs/recommendation_rules.json") -> None:
        self._path = Path(path)
        self._cache: Optional[List[RecommendationRule]] = None

    def _ensure(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text('{"rules":[]}', encoding="utf-8")

    def load(self) -> List[RecommendationRule]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {"rules": []}
        rules = [RecommendationRule(**r) for r in data.get("rules", [])]
        self._cache = rules
        return rules

    def save(self, rules: List[RecommendationRule]) -> None:
        data = {"rules": [r.model_dump() for r in rules]}
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._cache = rules

    def get_cached(self) -> List[RecommendationRule]:
        if self._cache is None:
            return self.load()
        return self._cache

    def evaluate(self, row: Dict[str, Any]) -> Optional[RecommendationRule]:
        """返回第一条命中规则（按 priority 降序，仅 enabled）。"""
        rules = sorted(self.get_cached(), key=lambda r: r.priority, reverse=True)
        app_id = str(row.get("app_id", ""))
        for rule in rules:
            if not rule.enabled:
                continue
            if rule.app_match != "*" and rule.app_match != app_id:
                continue
            if self._match_conditions(rule.conditions, row):
                return rule
        return None

    @staticmethod
    def _match_conditions(conditions: List[RuleCondition], row: Dict[str, Any]) -> bool:
        for c in conditions:
            val = row.get(c.field)
            if val is None:
                return False
            try:
                v = float(val)
                cv = float(c.value)
            except (ValueError, TypeError):
                v = str(val)
                cv = str(c.value)
            if c.op == ">=" and not (v >= cv):
                return False
            if c.op == "<=" and not (v <= cv):
                return False
            if c.op == ">" and not (v > cv):
                return False
            if c.op == "<" and not (v < cv):
                return False
            if c.op == "==" and not (v == cv):
                return False
            if c.op == "!=" and not (v != cv):
                return False
        return True

    def add(self, rule: RecommendationRule) -> RecommendationRule:
        rules = self.load()
        rules.append(rule)
        self.save(rules)
        return rule

    def update(self, rule_id: str, updates: Dict[str, Any]) -> Optional[RecommendationRule]:
        rules = self.load()
        for i, r in enumerate(rules):
            if r.id == rule_id:
                updated = r.model_copy(update=updates)
                rules[i] = updated
                self.save(rules)
                return updated
        return None

    def delete(self, rule_id: str) -> bool:
        rules = self.load()
        new_rules = [r for r in rules if r.id != rule_id]
        if len(new_rules) == len(rules):
            return False
        self.save(new_rules)
        return True
