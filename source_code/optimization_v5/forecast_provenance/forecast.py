from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


TARGETS = {
    "load": "load_kw",
    "pv": "pv_actual_kw",
    "price": "price_actual_yuan_per_kwh",
}

FEATURES = {
    "load": [
        "interval_index", "weekday", "month", "day_of_year", "time_sin", "time_cos",
        "typical_load_kw", "load_kw_lag1d", "load_kw_lag7d", "load_kw_hist_mean7d", "load_kw_hist_std7d",
    ],
    "pv": [
        "interval_index", "weekday", "month", "day_of_year", "time_sin", "time_cos",
        "typical_pv_kw", "pv_actual_kw_lag1d", "pv_actual_kw_lag7d", "pv_actual_kw_hist_mean7d",
        "pv_actual_kw_hist_std7d", "pv_hist_max28d",
    ],
    "price": [
        "interval_index", "weekday", "month", "day_of_year", "time_sin", "time_cos",
        "typical_price_yuan_per_kwh", "price_actual_yuan_per_kwh_lag1d", "price_actual_yuan_per_kwh_lag7d",
        "price_actual_yuan_per_kwh_hist_mean7d", "price_actual_yuan_per_kwh_hist_std7d",
        "load_kw_lag1d", "pv_actual_kw_lag1d", "net_load_kw_lag1d",
    ],
}


@dataclass
class ForecastArtifacts:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    training_log: pd.DataFrame


def _model(alpha: float, seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=90,
        learning_rate=0.06,
        num_leaves=23,
        min_child_samples=35,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.2,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def _pinball(y: np.ndarray, pred: np.ndarray, alpha: float) -> float:
    error = y - pred
    return float(np.mean(np.maximum(alpha * error, (alpha - 1.0) * error)))


def build_forecasts(actual: pd.DataFrame, seed: int = 42) -> ForecastArtifacts:
    frame = actual.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["interval_start"] = pd.to_datetime(frame["interval_start"])
    prediction_parts: list[pd.DataFrame] = []
    logs: list[dict[str, object]] = []
    month_starts = pd.date_range("2025-02-01", "2025-12-01", freq="MS")
    for month_start in month_starts:
        next_month = month_start + pd.offsets.MonthBegin(1)
        train_pool = frame[frame["date"] < month_start]
        calibration_start = month_start - pd.Timedelta(days=7)
        fit = train_pool[train_pool["date"] < calibration_start]
        calibration = train_pool[train_pool["date"] >= calibration_start]
        target_rows = frame[(frame["date"] >= month_start) & (frame["date"] < next_month)].copy()
        output = target_rows[["date", "interval_index", "interval_start", "load_kw", "pv_actual_kw", "price_actual_yuan_per_kwh"]].copy()
        for target_name, target_column in TARGETS.items():
            features = FEATURES[target_name]
            x_fit = fit[features]
            y_fit = fit[target_column].to_numpy(dtype=float)
            x_cal = calibration[features]
            y_cal = calibration[target_column].to_numpy(dtype=float)
            x_pred = target_rows[features]
            raw: dict[float, np.ndarray] = {}
            cal_pred: dict[float, np.ndarray] = {}
            for alpha in (0.1, 0.5, 0.9):
                model = _model(alpha, seed + int(alpha * 100) + month_start.month)
                model.fit(x_fit, y_fit)
                raw[alpha] = model.predict(x_pred)
                cal_pred[alpha] = model.predict(x_cal)
            nonconformity = np.maximum(cal_pred[0.1] - y_cal, y_cal - cal_pred[0.9])
            conformal_delta = max(0.0, float(np.quantile(nonconformity, 0.9, method="higher")))
            lower = raw[0.1] - conformal_delta
            median = raw[0.5]
            upper = raw[0.9] + conformal_delta
            stacked = np.sort(np.vstack([lower, median, upper]), axis=0)
            if target_name in {"load", "pv"}:
                stacked = np.maximum(stacked, 0.0)
            else:
                stacked = np.maximum(stacked, 0.001)
            output[f"{target_name}_q10"] = stacked[0]
            output[f"{target_name}_q50"] = stacked[1]
            output[f"{target_name}_q90"] = stacked[2]
            logs.append(
                {
                    "forecast_month": month_start.strftime("%Y-%m"),
                    "target": target_name,
                    "fit_end": (calibration_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "calibration_window": f"{calibration_start:%Y-%m-%d}..{month_start - pd.Timedelta(days=1):%Y-%m-%d}",
                    "fit_rows": len(fit),
                    "calibration_rows": len(calibration),
                    "conformal_delta": conformal_delta,
                }
            )
        prediction_parts.append(output)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(["date", "interval_index"])
    metric_rows: list[dict[str, object]] = []
    for name, actual_column in TARGETS.items():
        y = predictions[actual_column].to_numpy(dtype=float)
        q10 = predictions[f"{name}_q10"].to_numpy(dtype=float)
        q50 = predictions[f"{name}_q50"].to_numpy(dtype=float)
        q90 = predictions[f"{name}_q90"].to_numpy(dtype=float)
        lag_column = f"{actual_column}_lag7d"
        lag_values = frame.set_index(["date", "interval_index"]).loc[
            pd.MultiIndex.from_frame(predictions[["date", "interval_index"]]), lag_column
        ].to_numpy(dtype=float)
        for model_name, pred in [("D-7 baseline", lag_values), ("LightGBM q50", q50)]:
            metric_rows.append(
                {
                    "target": name,
                    "model": model_name,
                    "mae": float(np.nanmean(np.abs(y - pred))),
                    "rmse": float(np.sqrt(np.nanmean((y - pred) ** 2))),
                    "pinball_q50": _pinball(y[np.isfinite(pred)], pred[np.isfinite(pred)], 0.5),
                    "q10_q90_coverage": np.nan if model_name.startswith("D-") else float(np.mean((y >= q10) & (y <= q90))),
                }
            )
    return ForecastArtifacts(predictions.reset_index(drop=True), pd.DataFrame(metric_rows), pd.DataFrame(logs))


def load_or_build_forecasts(actual: pd.DataFrame, cache_path: Path, seed: int = 42, force: bool = False) -> ForecastArtifacts:
    metrics_path = cache_path.with_name("forecast_metrics.csv")
    log_path = cache_path.with_name("forecast_training_log.csv")
    if cache_path.exists() and metrics_path.exists() and log_path.exists() and not force:
        return ForecastArtifacts(
            pd.read_csv(cache_path, parse_dates=["date", "interval_start"]),
            pd.read_csv(metrics_path),
            pd.read_csv(log_path),
        )
    artifacts = build_forecasts(actual, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.predictions.to_csv(cache_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    artifacts.metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    artifacts.training_log.to_csv(log_path, index=False, encoding="utf-8-sig")
    return artifacts

