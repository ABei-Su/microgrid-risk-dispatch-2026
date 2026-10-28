from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import ModelConfig


class ScenarioFactory:
    def __init__(
        self,
        actual: pd.DataFrame,
        predictions: pd.DataFrame,
        official_forecast: pd.DataFrame,
        cfg: ModelConfig,
    ) -> None:
        self.cfg = cfg
        self.actual = actual.copy()
        self.actual["date"] = pd.to_datetime(self.actual["date"]).dt.normalize()
        self.actual["interval_start"] = pd.to_datetime(self.actual["interval_start"])
        self.predictions = predictions.copy()
        self.predictions["date"] = pd.to_datetime(self.predictions["date"]).dt.normalize()
        self.predictions["interval_start"] = pd.to_datetime(self.predictions["interval_start"])
        self.official = official_forecast.copy()
        self.official["issue_date"] = pd.to_datetime(self.official["issue_date"]).dt.normalize()
        self.official["issue_timestamp"] = pd.to_datetime(self.official["issue_timestamp"])
        self.official["target_timestamp"] = pd.to_datetime(self.official["target_timestamp"])
        actual_lookup = self.actual.set_index("interval_start")["pv_actual_kw"]
        self.official["actual_pv_kw"] = self.official["target_timestamp"].map(actual_lookup)
        self.official["error_kw"] = self.official["actual_pv_kw"] - self.official["pv_forecast_kw"]
        self.actual_by_date = {d: g.sort_values("interval_index") for d, g in self.actual.groupby("date")}
        self.pred_by_date = {d: g.sort_values("interval_index") for d, g in self.predictions.groupby("date")}

    def _rng(self, date: pd.Timestamp, tag: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{date:%Y-%m-%d}|{tag}|{self.cfg.seed}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _history_dates(self, date: pd.Timestamp) -> list[pd.Timestamp]:
        lower = date - pd.Timedelta(days=self.cfg.residual_window_days)
        return [d for d in self.actual_by_date if lower <= d < date and d >= pd.Timestamp("2025-01-08")]

    def _residual_matrix(self, dates: list[pd.Timestamp], column: str) -> np.ndarray:
        rows = []
        for d in dates:
            frame = self.actual_by_date[d]
            lag1 = frame[f"{column}_lag1d"].to_numpy(float)
            lag7 = frame[f"{column}_lag7d"].to_numpy(float)
            baseline = np.where(np.isfinite(lag7), 0.65 * lag7 + 0.35 * lag1, lag1)
            rows.append(frame[column].to_numpy(float) - baseline)
        matrix = np.vstack(rows)
        return matrix - np.nanmedian(matrix, axis=0, keepdims=True)

    def standard_scenarios(
        self,
        date: pd.Timestamp,
        start_index: int = 0,
        load_bias_kw: float = 0.0,
        include_variable_price: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        date = pd.Timestamp(date).normalize()
        pred = self.pred_by_date[date]
        idx = np.arange(start_index, 144)
        load_base = pred["load_q50"].to_numpy(float)[idx] + load_bias_kw
        pv_base = pred["pv_q50"].to_numpy(float)[idx]
        price_base = pred["price_q50"].to_numpy(float)[idx] if include_variable_price else self.actual_by_date[date]["typical_price_yuan_per_kwh"].to_numpy(float)[idx]
        history_dates = self._history_dates(date)
        load_res = self._residual_matrix(history_dates, "load_kw")[:, idx]
        pv_res = self._residual_matrix(history_dates, "pv_actual_kw")[:, idx]
        price_res = self._residual_matrix(history_dates, "price_actual_yuan_per_kwh")[:, idx]
        rng = self._rng(date, f"standard-{start_index}-{include_variable_price}")
        sample_indices = rng.choice(len(history_dates), size=max(1, self.cfg.scenario_count - 1), replace=len(history_dates) < self.cfg.scenario_count)
        loads = [np.maximum(load_base + load_res[i], 0.0) for i in sample_indices]
        pvs = [np.maximum(pv_base + pv_res[i], 0.0) for i in sample_indices]
        prices = [np.maximum(price_base + price_res[i], 0.001) if include_variable_price else price_base.copy() for i in sample_indices]
        loads.append(pred["load_q90"].to_numpy(float)[idx] + load_bias_kw)
        pvs.append(np.maximum(pred["pv_q10"].to_numpy(float)[idx], 0.0))
        prices.append(np.maximum(pred["price_q90"].to_numpy(float)[idx], 0.001) if include_variable_price else price_base.copy())
        return np.vstack(prices), np.vstack(loads), np.vstack(pvs)

    def official_scenarios(
        self,
        date: pd.Timestamp,
        issue_time: str,
        start_index: int,
        load_bias_kw: float = 0.0,
        include_variable_price: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        date = pd.Timestamp(date).normalize()
        actual_day = self.actual_by_date[date]
        targets = actual_day["interval_start"].to_numpy()[start_index:]
        current = self.official[(self.official["issue_date"] == date) & (self.official["issue_time"] == issue_time)]
        forecast_map = current.set_index("target_timestamp")["pv_forecast_kw"]
        pv_base = pd.Series(pd.to_datetime(targets)).map(forecast_map).to_numpy(float)
        if np.isnan(pv_base).any():
            raise ValueError(f"official forecast alignment failed for {date:%Y-%m-%d} {issue_time}")
        pred = self.pred_by_date[date]
        idx = np.arange(start_index, 144)
        load_base = pred["load_q50"].to_numpy(float)[idx] + load_bias_kw
        price_base = pred["price_q50"].to_numpy(float)[idx] if include_variable_price else actual_day["typical_price_yuan_per_kwh"].to_numpy(float)[idx]
        history_dates = self._history_dates(date)
        load_res = self._residual_matrix(history_dates, "load_kw")[:, idx]
        price_res = self._residual_matrix(history_dates, "price_actual_yuan_per_kwh")[:, idx]
        historical = self.official[(self.official["issue_time"] == issue_time) & (self.official["issue_date"] < date)]
        historical = historical[historical["issue_date"].isin(history_dates)]
        rng = self._rng(date, f"official-{issue_time}-{include_variable_price}")
        sample_dates = rng.choice(history_dates, size=max(1, self.cfg.scenario_count - 1), replace=len(history_dates) < self.cfg.scenario_count)
        loads: list[np.ndarray] = []; pvs: list[np.ndarray] = []; prices: list[np.ndarray] = []
        history_position = {d: i for i, d in enumerate(history_dates)}
        current_issue = pd.Timestamp(date) + pd.Timedelta(hours=int(issue_time.split(":")[0]))
        lead_steps = ((pd.to_datetime(targets) - current_issue) / pd.Timedelta(minutes=10)).astype(int)
        for sampled_date in sample_dates:
            i = history_position[pd.Timestamp(sampled_date)]
            sampled_issue = pd.Timestamp(sampled_date) + pd.Timedelta(hours=int(issue_time.split(":")[0]))
            sampled_targets = sampled_issue + pd.to_timedelta(lead_steps * 10, unit="min")
            key = self.official[(self.official["issue_date"] == pd.Timestamp(sampled_date)) & (self.official["issue_time"] == issue_time)].set_index("target_timestamp")["error_kw"]
            pv_error = pd.Series(sampled_targets).map(key).fillna(0.0).to_numpy(float)
            loads.append(np.maximum(load_base + load_res[i], 0.0))
            pvs.append(np.maximum(pv_base + pv_error, 0.0))
            prices.append(np.maximum(price_base + price_res[i], 0.001) if include_variable_price else price_base.copy())
        loads.append(pred["load_q90"].to_numpy(float)[idx] + load_bias_kw)
        if len(historical):
            by_lead = (
                historical.assign(
                    lead_step=(
                        (historical["target_timestamp"] - historical["issue_timestamp"])
                        / pd.Timedelta(minutes=10)
                    ).astype(int)
                )
                .groupby("lead_step")["error_kw"]
                .quantile(0.1)
            )
            conservative_error = pd.Series(lead_steps).map(by_lead).fillna(0.0).to_numpy(float)
        else:
            conservative_error = np.zeros(len(lead_steps), dtype=float)
        pvs.append(np.maximum(pv_base + conservative_error, 0.0))
        prices.append(np.maximum(pred["price_q90"].to_numpy(float)[idx], 0.001) if include_variable_price else price_base.copy())
        return np.vstack(prices), np.vstack(loads), np.vstack(pvs)
