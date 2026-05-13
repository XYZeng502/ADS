#!/usr/bin/env python3
"""
Merge legacy granular daily CSV with per-day exports from download_daily.

API: GET {endpoint}?date=YYYY-MM-DD returns JSON {"url": "<zip url>"}.
Each zip contains one CSV (often spanning multiple calendar days and many 应用ID).

Merge semantics (upsert by 应用ID + 日期):
  - Each ZIP is read in full (all rows, all dates in the file).
  - Let K = distinct (应用ID, 日期) in that file. Remove from the accumulated table every row
    whose (应用ID, 日期) is in K, then append the entire new file (reindexed to canonical columns).
  - Effect: existing apps get same-calendar-day (and any other dates present in the ZIP) replaced
    by the newer export; brand-new 应用ID rows are appended as-is, including their history rows
    inside the ZIP.
  - Process request dates in ascending order so later daily pulls overwrite earlier ones for the
    same (应用ID, 日期) when both files contain that key.

Finally sort like legacy snapshots: 应用ID, 日期 ascending; 消耗金额 descending; remaining columns ascending.

When --snapshot-date is set: skip legacy and incremental merge; download that date once and treat the ZIP CSV
as the full dataset (e.g. 2026-05-11 export is already all history + updates).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

DEFAULT_LEGACY = Path(__file__).resolve().parents[1] / "daily_20260421_120112.csv"
DEFAULT_ENDPOINT = "http://123.56.3.39:41108/download_daily"


def _http_get(url: str, timeout: int) -> bytes:
    req = Request(url, headers={"User-Agent": "ad_ml-merge_daily_downloads/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_download_daily_json(raw: bytes) -> dict:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"响应不是合法 JSON: {e}") from e
    if not isinstance(obj, dict) or "url" not in obj:
        raise ValueError(f"JSON 缺少 url 字段: {repr(obj)[:500]}")
    return obj


def _read_csv_from_zip(data: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"ZIP 内期望单个 CSV，实际: {names}")
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh)


def _daterange(d0: date, d1: date) -> Iterable[date]:
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def _canonical_columns_for_legacy(legacy_cols: list[str]) -> list[str]:
    """Insert API-only dimensions after 广告主ID so legacy rows get NA until口径补齐."""
    cols = list(legacy_cols)
    if "广告主ID" not in cols:
        raise ValueError("legacy CSV 缺少列 广告主ID，无法对齐增量字段")
    i = cols.index("广告主ID") + 1
    insert_after_advertiser = ["集团ID"]
    for name in insert_after_advertiser:
        if name not in cols:
            cols = cols[:i] + [name] + cols[i:]
            i += 1
    return cols


def _extend_canonical(canonical: list[str], df_cols: Iterable[str]) -> list[str]:
    """追加远端新增的列（排在末尾），并同步回填已有 DataFrame。"""
    canon = list(canonical)
    seen = set(canon)
    for c in df_cols:
        if c not in seen:
            canon.append(c)
            seen.add(c)
    return canon


def _upsert_by_app_date(
    base: pd.DataFrame, new_df: pd.DataFrame, canonical: list[str]
) -> pd.DataFrame:
    """Drop base rows in (应用ID,日期) keys present in new_df, then append new_df (canonical columns)."""
    new_df = new_df.copy()
    new_df["日期"] = pd.to_datetime(new_df["日期"]).dt.strftime("%Y-%m-%d")
    new_aligned = new_df.reindex(columns=canonical)
    if new_aligned.empty:
        return base.reindex(columns=canonical)

    keys = new_aligned[["应用ID", "日期"]].drop_duplicates()
    if len(base) > 0:
        tagged = base.merge(keys.assign(__purge=1), on=["应用ID", "日期"], how="left")
        base = tagged[tagged["__purge"].isna()].drop(columns=["__purge"])
    return pd.concat([base, new_aligned], ignore_index=True)


def _legacy_sort_columns(columns: list[str]) -> tuple[list[str], list[bool]]:
    """Match legacy snapshot ordering: app/date ascending, spend descending, rest ascending."""
    keys: list[str] = []
    asc: list[bool] = []
    if "应用ID" in columns:
        keys.append("应用ID")
        asc.append(True)
    if "日期" in columns:
        keys.append("日期")
        asc.append(True)
    if "消耗金额" in columns:
        keys.append("消耗金额")
        asc.append(False)
    tail = [
        c
        for c in columns
        if c not in keys and c != "消耗金额"
    ]
    keys.extend(tail)
    asc.extend([True] * len(tail))
    return keys, asc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="download_daily：增量 upsert 合并 legacy，或单日全量快照（--snapshot-date）"
    )
    parser.add_argument(
        "--snapshot-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="仅下载该日 ZIP，整包作为全量结果写出（不读 legacy、不多日合并）。例如服务端 5/11 即为全量。",
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=DEFAULT_LEGACY,
        help="历史全量 CSV（默认仓库根目录 daily_20260421_120112.csv）；--snapshot-date 时忽略",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="download_daily 前缀 URL（不含查询串）",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="起始日期 YYYY-MM-DD；默认 legacy 最大日期次日",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="结束日期 YYYY-MM-DD（含）；默认今天（本地日历）",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出合并 CSV 路径")
    parser.add_argument("--timeout", type=int, default=120, help="单次 HTTP 超时秒数")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="某日无数据（404 等）时跳过并继续；默认遇到错误直接退出",
    )
    args = parser.parse_args()

    if args.snapshot_date:
        snap = datetime.strptime(args.snapshot_date, "%Y-%m-%d").date()
        ds = snap.strftime("%Y-%m-%d")
        endpoint = args.endpoint.rstrip("/")
        meta_url = f"{endpoint}?date={ds}"
        try:
            meta_raw = _http_get(meta_url, timeout=args.timeout)
            payload = _parse_download_daily_json(meta_raw)
            zip_bytes = _http_get(str(payload["url"]), timeout=args.timeout)
            merged = _read_csv_from_zip(zip_bytes)
        except (HTTPError, URLError, TimeoutError, ValueError, zipfile.BadZipFile, KeyError) as e:
            print(f"[失败] 全量快照 {ds} {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        merged["日期"] = pd.to_datetime(merged["日期"]).dt.strftime("%Y-%m-%d")
        keys, ascending = _legacy_sort_columns(list(merged.columns))
        merged = merged.sort_values(keys, ascending=ascending, kind="mergesort").reset_index(drop=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(
            f"[全量快照] {ds} 写入 {args.output} 行数={len(merged)} "
            f"日期范围 {merged['日期'].min()} ~ {merged['日期'].max()}"
        )
        return 0

    legacy_path: Path = args.legacy
    if not legacy_path.is_file():
        print(f"找不到 legacy 文件: {legacy_path}", file=sys.stderr)
        return 1

    legacy = pd.read_csv(legacy_path)
    if legacy.empty:
        print("legacy CSV 为空", file=sys.stderr)
        return 1

    legacy["日期"] = pd.to_datetime(legacy["日期"]).dt.strftime("%Y-%m-%d")
    max_legacy_day = pd.to_datetime(legacy["日期"]).max().date()

    end_day = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    if args.start:
        start_day = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_day = max_legacy_day + timedelta(days=1)

    canonical = _canonical_columns_for_legacy(list(legacy.columns))

    if start_day > end_day:
        merged = legacy.reindex(columns=canonical)
        print(f"无需下载：起始 {start_day} 晚于结束 {end_day}，仅写出 legacy 排序副本")
    else:
        merged = legacy.reindex(columns=canonical)
        endpoint = args.endpoint.rstrip("/")

        for d in _daterange(start_day, end_day):
            ds = d.strftime("%Y-%m-%d")
            meta_url = f"{endpoint}?date={ds}"
            try:
                meta_raw = _http_get(meta_url, timeout=args.timeout)
                payload = _parse_download_daily_json(meta_raw)
                zip_url = str(payload["url"])
                zip_bytes = _http_get(zip_url, timeout=args.timeout)
                chunk_df = _read_csv_from_zip(zip_bytes)
            except HTTPError as e:
                if e.code == 404 and args.skip_missing:
                    print(f"[跳过] {ds} HTTP 404", file=sys.stderr)
                    continue
                print(f"[失败] {ds} meta_url={meta_url} HTTPError {e}", file=sys.stderr)
                return 1
            except (URLError, TimeoutError, ValueError, zipfile.BadZipFile, KeyError) as e:
                if args.skip_missing:
                    print(f"[跳过] {ds} {type(e).__name__}: {e}", file=sys.stderr)
                    continue
                print(f"[失败] {ds} {type(e).__name__}: {e}", file=sys.stderr)
                return 1

            extended = _extend_canonical(canonical, chunk_df.columns)
            if extended != canonical:
                canonical = extended
                merged = merged.reindex(columns=canonical)

            n_chunk = len(chunk_df)
            n_keys = (
                chunk_df.assign(
                    __d=pd.to_datetime(chunk_df["日期"]).dt.strftime("%Y-%m-%d")
                )[["应用ID", "__d"]]
                .drop_duplicates()
                .rename(columns={"__d": "日期"})
            )
            merged = _upsert_by_app_date(merged, chunk_df, canonical)
            print(
                f"[OK] {ds} zip_rows={n_chunk} distinct(应用ID,日期)={len(n_keys)} "
                f"merged_total={len(merged)}"
            )

    keys, ascending = _legacy_sort_columns(list(merged.columns))
    merged = merged.sort_values(keys, ascending=ascending, kind="mergesort").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        f"写入 {args.output} 行数={len(merged)} 日期范围 "
        f"{merged['日期'].min()} ~ {merged['日期'].max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
