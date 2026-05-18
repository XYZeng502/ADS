"""日历健康检查：验证 business calendar 覆盖度与有效性。"""
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from app.core.calendar import CalendarService


@dataclass
class CalendarHealth:
    file_exists: bool
    version: str = ""
    can_parse: bool = False
    holiday_count: int = 0
    first_date: str | None = None
    last_date: str | None = None
    covers_data_dates: bool = False
    covers_future_30d: bool = False
    missing_dates: list[str] = field(default_factory=list)
    status: str = "unknown"


def check_calendar_health(
    calendar_path: Path | None = None,
    data_csv: Path | None = None,
) -> CalendarHealth:
    if calendar_path is None:
        calendar_path = Path("data/business_calendar_2026.json")
    if data_csv is None:
        data_csv = Path("daily_merged.csv")

    ch = CalendarHealth(file_exists=calendar_path.exists())

    if not ch.file_exists:
        ch.status = "error"
        ch.missing_dates = ["calendar file not found"]
        return ch

    # 复用 CalendarService 解析日历
    try:
        cal = CalendarService(calendar_path)
        ch.version = cal.version
        ch.can_parse = True
        windows = cal._holiday_windows
        ch.holiday_count = len(windows)
        if windows:
            starts = sorted(w.start.isoformat() for w in windows)
            ends = sorted(w.end.isoformat() for w in windows)
            ch.first_date = starts[0]
            ch.last_date = ends[-1]
    except Exception:
        ch.status = "error"
        ch.missing_dates = ["calendar parse error"]
        return ch

    holiday_days = cal._holiday_days()
    cal_years = {d.year for d in holiday_days}

    # 检查 CSV 数据日期是否同日历年
    if data_csv.exists():
        csv_years: set[int] = set()
        try:
            with data_csv.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = row.get("日期", "").strip()
                    if d:
                        try:
                            csv_years.add(datetime.strptime(d, "%Y-%m-%d").date().year)
                        except ValueError:
                            pass
        except Exception:
            pass

        uncovered_years = csv_years - cal_years
        ch.covers_data_dates = len(uncovered_years) == 0
        if uncovered_years:
            ch.missing_dates.append(f"CSV contains years not in calendar: {sorted(uncovered_years)}")

    # 检查未来 30 天覆盖
    today = date.today()
    future_cutoff = today + timedelta(days=30)
    if ch.last_date:
        last_cal_date = datetime.strptime(ch.last_date, "%Y-%m-%d").date()
        ch.covers_future_30d = last_cal_date >= future_cutoff
    if not ch.covers_future_30d:
        ch.missing_dates.append(f"Calendar last date ({ch.last_date}) < future 30d ({future_cutoff.isoformat()})")

    if not ch.covers_data_dates or not ch.covers_future_30d:
        ch.status = "warning"
    elif ch.holiday_count == 0:
        ch.status = "warning"
    else:
        ch.status = "ok"

    return ch
