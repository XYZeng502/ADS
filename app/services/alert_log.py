"""告警持久化：JSONL 追加，保留 30 天。"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from app.schemas import RiskAlert

_PATH = Path("outputs/alert_history.jsonl")
_MAX_DAYS = 30


class AlertLog:
    @classmethod
    def append(cls, alerts: list[RiskAlert]) -> None:
        if not alerts:
            return
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # 读今天已有告警去重
        today_keys = set()
        if _PATH.exists():
            for line in _PATH.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line.strip())
                    if r.get("date") == today:
                        today_keys.add((r.get("category", ""), r.get("product_id")))
                except json.JSONDecodeError:
                    continue

        _PATH.parent.mkdir(parents=True, exist_ok=True)
        new_lines = []
        for a in alerts:
            key = (a.category, a.product_id)
            if key in today_keys:
                continue
            new_lines.append(json.dumps({
                "timestamp": now.isoformat(),
                "date": today,
                "level": a.level,
                "category": a.category,
                "message": a.message,
                "product_id": a.product_id,
            }, ensure_ascii=False))
            today_keys.add(key)

        if new_lines:
            with _PATH.open("a", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")

        cls._prune()

    @classmethod
    def query(cls, days: int = 7, level: Optional[str] = None) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        records = cls._load_all()
        result = [r for r in records if r.get("timestamp", "") >= cutoff]
        if level:
            result = [r for r in result if r.get("level") == level]
        result.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return result

    @classmethod
    def summary(cls, days: int = 7) -> dict:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        records = [r for r in cls._load_all() if r.get("timestamp", "") >= cutoff]
        by_level: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for r in records:
            by_level[r.get("level", "?")] = by_level.get(r.get("level", "?"), 0) + 1
            by_category[r.get("category", "?")] = by_category.get(r.get("category", "?"), 0) + 1
        return {
            "total": len(records),
            "days": days,
            "by_level": by_level,
            "by_category": by_category,
            "latest_alert": records[-1] if records else None,
        }

    @classmethod
    def _load_all(cls) -> list[dict]:
        if not _PATH.exists():
            return []
        records = []
        for line in _PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    @classmethod
    def _prune(cls) -> None:
        if not _PATH.exists():
            return
        cutoff = (datetime.now(timezone.utc) - timedelta(days=_MAX_DAYS)).isoformat()
        lines = []
        for line in _PATH.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line.strip())
                if r.get("timestamp", "") >= cutoff.isoformat():
                    lines.append(line.strip())
            except json.JSONDecodeError:
                continue
        _PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
