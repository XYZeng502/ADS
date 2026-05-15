import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

from chinese_calendar import (
    is_holiday as _cn_is_holiday,
    is_in_lieu as _cn_is_in_lieu,
    is_workday as _cn_is_workday,
)

from app.schemas import DayType


@dataclass(frozen=True)
class HolidayWindow:
    id: int
    name: str
    display_name: str
    start: date
    end: date


class CalendarService:
    """
    统一业务日历服务。

    数据源来自本地版本化日历文件，后续可由官方/API 数据源定期刷新；
    模型、求解器、规则引擎都应依赖这里生成的稳定日历口径。
    """

    def __init__(self, calendar_path: str | Path = "data/business_calendar_2026.json") -> None:
        self.calendar_path = Path(calendar_path)
        raw = self._load_calendar(self.calendar_path)
        self.version = str(raw.get("version", "unknown"))
        self.source = str(raw.get("source", "local"))
        self._holiday_windows = [
            HolidayWindow(
                id=int(x["id"]),
                name=str(x["name"]),
                display_name=str(x.get("display_name") or x["name"]),
                start=date.fromisoformat(str(x["start"])),
                end=date.fromisoformat(str(x["end"])),
            )
            for x in raw.get("holiday_windows", [])
        ]
        self._adjusted_workdays = {
            date.fromisoformat(str(x))
            for x in raw.get("adjusted_workdays", [])
        }
        self._shopping_days = {
            date.fromisoformat(str(x["date"]))
            for x in raw.get("shopping_days", [])
        }

    @staticmethod
    def _load_calendar(calendar_path: Path) -> dict:
        if calendar_path.exists():
            return json.loads(calendar_path.read_text(encoding="utf-8"))
        return {"holiday_windows": [], "adjusted_workdays": [], "shopping_days": []}

    @staticmethod
    def _is_summer_winter(day: date) -> bool:
        # 游戏投放常见长周期窗口：寒暑假
        return day.month in (1, 2, 7, 8)

    def holiday_window(self, day: date) -> Optional[HolidayWindow]:
        for window in self._holiday_windows:
            if window.start <= day <= window.end:
                return window
        return None

    def is_holiday(self, day: date) -> bool:
        # 法定假日窗口内（JSON 定义的假期区间）
        return self.holiday_window(day) is not None

    def is_adjusted_workday(self, day: date) -> bool:
        # 调休上班日：周末/假期但 chinese_calendar 标记为工作日
        return (day.weekday() >= 5 and _cn_is_workday(day)) or day in self._adjusted_workdays

    def is_rest_day(self, day: date) -> bool:
        # 任何休息日（周末+法定假日，排除调休上班）
        return _cn_is_holiday(day) and not self.is_adjusted_workday(day)

    def classify_day(self, day: date) -> DayType:
        # 优先级：调休上班 > 法定假日 > 电商节 > 寒暑假 > 周末 > 工作日
        if self.is_adjusted_workday(day):
            return "workday"
        if self.is_holiday(day):
            return "holiday"
        if day in self._shopping_days:
            return "shopping_festival"
        if self._is_summer_winter(day):
            return "summer_winter"
        if day.weekday() >= 5:
            return "weekend"
        return "workday"

    def get_scale_factors(self) -> Dict[DayType, float]:
        # 基于历史数据实证校准（加权 ROI 比）
        # weekend: 1.104 → 1.10, holiday: 1.152 → 1.15
        # shopping_festival / summer_winter 暂无数据覆盖，沿用预设值待验证
        return {
            "workday": 1.0,
            "weekend": 1.10,
            "holiday": 1.15,
            "shopping_festival": 1.15,
            "summer_winter": 1.10,
        }

    def days_to_next_holiday(self, day: date) -> int:
        holiday_days = self._holiday_days()
        future = [x for x in holiday_days if x >= day]
        if not future:
            return 99
        return min((x - day).days for x in future)

    def days_since_prev_holiday(self, day: date) -> int:
        holiday_days = self._holiday_days()
        past = [x for x in holiday_days if x <= day]
        if not past:
            return 99
        return min((day - x).days for x in past)

    def features_for_day(self, day: date) -> Dict[str, int | str]:
        window = self.holiday_window(day)
        holiday_id = window.id if window else 0
        holiday_seq_index = (day - window.start).days + 1 if window else 0
        holiday_days_remaining = (window.end - day).days if window else 0
        holiday_window_len = (window.end - window.start).days + 1 if window else 0
        return {
            "day_type": self.classify_day(day),
            "holiday_id": holiday_id,
            "holiday_name": window.name if window else "",
            "holiday_display_name": window.display_name if window else "",
            "is_holiday": int(window is not None),
            "is_rest_day": int(self.is_rest_day(day)),
            "is_adjusted_workday": int(self.is_adjusted_workday(day)),
            "is_weekend": int(day.weekday() >= 5),
            "holiday_seq_index": holiday_seq_index,
            "holiday_days_remaining": holiday_days_remaining,
            "holiday_window_len": holiday_window_len,
            "is_last_holiday_day": int(window is not None and day == window.end),
            "is_first_workday_after_holiday": int(
                _cn_is_workday(day) and self.is_holiday(day - timedelta(days=1))
            ),
            "days_to_next_holiday": min(self.days_to_next_holiday(day), 30),
            "days_since_prev_holiday": min(self.days_since_prev_holiday(day), 30),
        }

    def _holiday_days(self) -> list[date]:
        days: list[date] = []
        for window in self._holiday_windows:
            days.extend(window.start + timedelta(days=i) for i in range((window.end - window.start).days + 1))
        return sorted(days)
