"""数据管道：入口验证 + 健康检查。在预测脚本入口调用，拒绝不健康数据。"""
import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path


@dataclass
class DataHealth:
    csv_exists: bool
    last_date: str | None = None
    row_count: int = 0
    app_count: int = 0
    days_behind: int | None = None
    columns_ok: bool = True
    missing_cols: list[str] = field(default_factory=list)
    critical_null_rate: dict[str, float] = field(default_factory=dict)
    spend_drift_pct: float | None = None
    status: str = "unknown"  # "healthy" | "stale" | "error"

    @property
    def ok(self) -> bool:
        return self.status != "error"


_CRITICAL_COLS = ["消耗金额", "首日广告收入", "买量广告收入"]


def check_data_health(csv_path: Path) -> DataHealth:
    h = DataHealth(csv_exists=csv_path.exists())

    if not h.csv_exists:
        h.status = "error"
        h.missing_cols = ["file_not_found"]
        return h

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            header_set = set(header)

            # 关键列存在性（与 prediction_artifacts 口径一致）
            required = ["应用ID", "日期", "消耗金额", "首日广告收入", "买量广告收入"]
            h.missing_cols = [c for c in required if c not in header_set]
            h.columns_ok = len(h.missing_cols) == 0
            if not h.columns_ok:
                h.status = "error"
                return h

            # 遍历统计
            date_set: set[str] = set()
            app_set: set[str] = set()
            null_counts: dict[str, int] = {c: 0 for c in _CRITICAL_COLS}
            daily_spend: dict[str, float] = {}

            for row in reader:
                h.row_count += 1
                app_id = row.get("应用ID", "")
                day = row.get("日期", "")
                if app_id:
                    app_set.add(app_id)
                if day:
                    date_set.add(day)

                for col in _CRITICAL_COLS:
                    v = row.get(col, "")
                    if v == "" or v is None:
                        null_counts[col] += 1

                spend_str = row.get("消耗金额", "")
                if day and spend_str and spend_str.strip():
                    try:
                        daily_spend[day] = daily_spend.get(day, 0.0) + float(spend_str)
                    except ValueError:
                        pass

            h.app_count = len(app_set)
            sorted_dates = sorted(date_set)
            h.last_date = sorted_dates[-1] if sorted_dates else None

            if h.row_count > 0:
                for col in _CRITICAL_COLS:
                    h.critical_null_rate[col] = round(null_counts[col] / h.row_count, 4)

            # 新鲜度
            if h.last_date:
                try:
                    last_dt = datetime.strptime(h.last_date, "%Y-%m-%d").date()
                    h.days_behind = (date.today() - last_dt).days
                except ValueError:
                    pass

            # spend 日环比漂移（最近两天）
            if len(daily_spend) >= 2:
                last_two = sorted(daily_spend.keys())[-2:]
                prev = daily_spend.get(last_two[0], 0.0)
                curr = daily_spend.get(last_two[1], 0.0)
                if prev > 0:
                    h.spend_drift_pct = round((curr - prev) / prev * 100, 2)

            # 状态判定
            if h.columns_ok and h.days_behind is not None:
                h.status = "healthy" if h.days_behind <= 1 else "stale"
            else:
                h.status = "healthy"  # 列 OK 但无日期信息

    except Exception as e:
        h.status = "error"
        h.missing_cols = [str(e)]

    return h
