from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import read_actuals, read_forecast_hourly, read_typical_day


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson保形三次Hermite斜率，不依赖SciPy。"""
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros(n, dtype=float)
    if n == 2:
        slopes[:] = delta[0]
        return slopes
    for k in range(1, n - 1):
        if delta[k - 1] == 0 or delta[k] == 0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            slopes[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            slopes[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    slopes[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(slopes[0]) != np.sign(delta[0]):
        slopes[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(slopes[0]) > abs(3 * delta[0]):
        slopes[0] = 3 * delta[0]
    slopes[-1] = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(slopes[-1]) != np.sign(delta[-1]):
        slopes[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(slopes[-1]) > abs(3 * delta[-1]):
        slopes[-1] = 3 * delta[-1]
    return slopes


def pchip_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    slopes = _pchip_slopes(x, y)
    idx = np.clip(np.searchsorted(x, x_new, side="right") - 1, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (x_new - x[idx]) / h
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * y[idx] + h10 * h * slopes[idx] + h01 * y[idx + 1] + h11 * h * slopes[idx + 1]


def add_leakage_safe_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.sort_values(["date", "interval_index"]).copy()
    frame["pv_actual_kw"] = frame["pv_actual_kw"].clip(lower=0)
    frame["net_load_kw"] = frame["load_kw"] - frame["pv_actual_kw"]
    frame["load_kwh"] = frame["load_kw"] / 6.0
    frame["pv_actual_kwh"] = frame["pv_actual_kw"] / 6.0
    frame["net_load_kwh"] = frame["net_load_kw"] / 6.0
    frame["weekday"] = frame["date"].dt.weekday
    frame["month"] = frame["date"].dt.month
    frame["day_of_year"] = frame["date"].dt.dayofyear
    minutes = frame["interval_start"].dt.hour * 60 + frame["interval_start"].dt.minute
    angle = 2 * math.pi * minutes / (24 * 60)
    frame["time_sin"] = np.sin(angle)
    frame["time_cos"] = np.cos(angle)
    grouped = frame.groupby("interval_index", sort=False)
    for column in ["load_kw", "pv_actual_kw", "price_actual_yuan_per_kwh", "net_load_kw"]:
        frame[f"{column}_lag1d"] = grouped[column].shift(1)
        frame[f"{column}_lag7d"] = grouped[column].shift(7)
        frame[f"{column}_hist_mean7d"] = grouped[column].transform(
            lambda values: values.shift(1).rolling(7, min_periods=1).mean()
        )
        frame[f"{column}_hist_std7d"] = grouped[column].transform(
            lambda values: values.shift(1).rolling(7, min_periods=2).std()
        )
    frame["pv_hist_max28d"] = grouped["pv_actual_kw"].transform(
        lambda values: values.shift(1).rolling(28, min_periods=1).max()
    )
    frame["has_7d_history"] = (frame["date"] >= frame["date"].min() + pd.Timedelta(days=7)).astype(int)
    frame["evaluation_role"] = np.where(
        frame["date"] < pd.Timestamp("2025-02-01"), "initial_train", "rolling_backtest"
    )
    frame["static_fold"] = np.select(
        [
            frame["date"] <= pd.Timestamp("2025-09-30"),
            frame["date"] <= pd.Timestamp("2025-11-30"),
        ],
        ["train", "validation"],
        default="test",
    )
    return frame


def interpolate_forecasts(hourly: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    pv_lookup = actual.set_index("interval_start")["pv_actual_kw"].to_dict()
    rows: list[pd.DataFrame] = []
    for issue_ts, group in hourly.groupby("issue_timestamp", sort=True):
        group = group.sort_values("lead_hour")
        current_pv = float(pv_lookup.get(pd.Timestamp(issue_ts), 0.0))
        x = np.arange(0, 25, dtype=float)
        y = np.r_[current_pv, group["pv_forecast_kw"].to_numpy(dtype=float)]
        x_new = np.arange(1, 145, dtype=float) / 6.0
        interpolated = np.maximum(pchip_interpolate(x, y, x_new), 0.0)
        target = pd.Timestamp(issue_ts) + pd.to_timedelta(np.arange(1, 145) * 10, unit="min")
        rows.append(
            pd.DataFrame(
                {
                    "issue_date": pd.Timestamp(issue_ts).normalize(),
                    "issue_time": pd.Timestamp(issue_ts).strftime("%H:%M"),
                    "issue_timestamp": pd.Timestamp(issue_ts),
                    "lead_step_10min": np.arange(1, 145),
                    "lead_minutes": np.arange(1, 145) * 10,
                    "target_timestamp": target,
                    "pv_forecast_kw": interpolated,
                    "interpolation_method": "shape_preserving_pchip",
                    "anchor_pv_kw": current_pv,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_all(input_dir: Path) -> dict[str, pd.DataFrame]:
    typical = read_typical_day(input_dir / "附件" / "附件1.xlsx")
    actual_raw = read_actuals(input_dir)
    typical_lookup = typical.set_index("interval_index")[
        ["typical_price_yuan_per_kwh", "typical_load_kw", "typical_pv_kw"]
    ]
    actual = add_leakage_safe_features(actual_raw.join(typical_lookup, on="interval_index"))
    hourly = read_forecast_hourly(input_dir / "附件" / "附件3.xlsx")
    ten_minute = interpolate_forecasts(hourly, actual_raw)
    return {
        "typical_day": typical,
        "actual_10min": actual,
        "official_forecast_hourly": hourly,
        "official_forecast_10min": ten_minute,
    }

