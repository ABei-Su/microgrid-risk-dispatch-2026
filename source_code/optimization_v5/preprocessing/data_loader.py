from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


def _clock_minutes(value: object) -> tuple[int, bool]:
    if isinstance(value, time):
        return value.hour * 60 + value.minute, False
    text = str(value).strip()
    next_day = "+1" in text
    text = text.replace("+1", "")
    hour, minute = (int(part) for part in text.split(":")[:2])
    return hour * 60 + minute, next_day


def interval_label(value: object) -> str:
    minutes, next_day = _clock_minutes(value)
    end_total = minutes + 10
    start_hour, start_minute = divmod(minutes, 60)
    end_day = next_day or end_total >= 24 * 60
    end_total %= 24 * 60
    end_hour, end_minute = divmod(end_total, 60)
    start_suffix = "+1" if next_day else ""
    end_suffix = "+1" if end_day else ""
    return (
        f"{start_hour}:{start_minute:02d}{start_suffix}-"
        f"{end_hour}:{end_minute:02d}{end_suffix}"
    )


def timestamp_from_label(date_value: object, label: object) -> pd.Timestamp:
    base = pd.Timestamp(date_value).normalize()
    minutes, next_day = _clock_minutes(label)
    return base + pd.Timedelta(days=int(next_day), minutes=minutes)


def read_typical_day(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    raw.columns = ["time_label", "typical_price_yuan_per_kwh", "typical_load_kw", "typical_pv_kw"]
    raw["interval_index"] = range(len(raw))
    raw["interval_label"] = raw["time_label"].map(interval_label)
    raw["energy_hours"] = 1.0 / 6.0
    raw["typical_load_kwh"] = raw["typical_load_kw"] / 6.0
    raw["typical_pv_kwh"] = raw["typical_pv_kw"] / 6.0
    return raw


def _wide_sheet_to_long(path: Path, sheet_name: str, value_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    date_col = raw.columns[0]
    long = raw.melt(id_vars=[date_col], var_name="time_label", value_name=value_name)
    long = long.rename(columns={date_col: "date"})
    time_order = {label: idx for idx, label in enumerate(raw.columns[1:])}
    long["interval_index"] = long["time_label"].map(time_order).astype(int)
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    long["interval_start"] = [
        timestamp_from_label(date_value, label)
        for date_value, label in zip(long["date"], long["time_label"])
    ]
    long["interval_end"] = long["interval_start"] + pd.Timedelta(minutes=10)
    long["interval_label"] = long["time_label"].map(interval_label)
    return long.sort_values(["date", "interval_index"]).reset_index(drop=True)


def read_actuals(input_dir: Path) -> pd.DataFrame:
    attachment_dir = input_dir / "附件"
    load = _wide_sheet_to_long(attachment_dir / "附件2.xlsx", "小区负载", "load_kw")
    pv = _wide_sheet_to_long(attachment_dir / "附件2.xlsx", "光伏发电实际功率", "pv_actual_kw")
    price = _wide_sheet_to_long(attachment_dir / "附件4.xlsx", "Sheet1", "price_actual_yuan_per_kwh")
    keys = ["date", "interval_index", "interval_start", "interval_end", "interval_label"]
    result = load[keys + ["load_kw"]].merge(
        pv[keys + ["pv_actual_kw"]], on=keys, validate="one_to_one"
    )
    result = result.merge(
        price[keys + ["price_actual_yuan_per_kwh"]], on=keys, validate="one_to_one"
    )
    return result.sort_values(["date", "interval_index"]).reset_index(drop=True)


def read_forecast_hourly(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    raw["日期"] = raw["日期"].replace("", pd.NA).ffill()
    raw["日期"] = pd.to_datetime(raw["日期"])
    records: list[dict[str, object]] = []
    forecast_columns: Iterable[str] = raw.columns[2:]
    for _, row in raw.iterrows():
        issue_clock = datetime.strptime(str(row["预报时刻"]), "%H:%M").time()
        issue_ts = pd.Timestamp.combine(row["日期"].date(), issue_clock)
        for lead, column in enumerate(forecast_columns, start=1):
            records.append(
                {
                    "issue_date": issue_ts.normalize(),
                    "issue_time": issue_clock.strftime("%H:%M"),
                    "issue_timestamp": issue_ts,
                    "lead_hour": lead,
                    "target_timestamp": issue_ts + pd.Timedelta(hours=lead),
                    "pv_forecast_kw": float(row[column]),
                }
            )
    return pd.DataFrame.from_records(records)


def find_result_templates(input_dir: Path) -> list[Path]:
    return sorted((input_dir / "附件" / "附件5").glob("result*.xlsx"))

