from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linprog as _linprog
import warnings
from scipy.optimize import OptimizeWarning

def linprog(*args, **kwargs):
    opts = dict(kwargs.pop("options", {})); opts["threads"] = 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=OptimizeWarning)
        return _linprog(*args, options=opts, **kwargs)
from scipy.sparse import coo_matrix, vstack

from candidates import ModelConfig


@dataclass
class PlanSolution:
    grid_kwh: np.ndarray
    expected_soc_kwh: np.ndarray
    objective_yuan: float
    runtime_seconds: float
    status: str
    iterations: int
    max_equality_residual: float
    initial_soc_value: float = 0.0


def solve_q1(price: np.ndarray, load_kw: np.ndarray, pv_kw: np.ndarray, cfg: ModelConfig) -> dict[str, object]:
    n = len(price)
    g0, c0, d0, w0, e0 = 0, n, 2 * n, 3 * n, 4 * n
    nvar = 5 * n
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    row = 0
    for t in range(n):
        for col, value in [(g0 + t, 1.0), (d0 + t, 1.0), (c0 + t, -1.0), (w0 + t, -1.0)]:
            rows.append(row); cols.append(col); vals.append(value)
        rhs.append(cfg.delta_hours * (load_kw[t] - pv_kw[t])); row += 1
    for t in range(n):
        for col, value in [(e0 + t, 1.0), (c0 + t, -cfg.eta_charge), (d0 + t, 1.0 / cfg.eta_discharge)]:
            rows.append(row); cols.append(col); vals.append(value)
        if t:
            rows.append(row); cols.append(e0 + t - 1); vals.append(-1.0)
            rhs.append(0.0)
        else:
            rhs.append(cfg.initial_soc_kwh)
        row += 1
    rows.append(row); cols.append(e0 + n - 1); vals.append(1.0); rhs.append(cfg.initial_soc_kwh)
    aeq = coo_matrix((vals, (rows, cols)), shape=(row + 1, nvar)).tocsr()
    bounds = (
        [(0.0, None)] * n
        + [(0.0, cfg.interval_power_limit_kwh)] * n
        + [(0.0, cfg.interval_power_limit_kwh)] * n
        + [(0.0, None)] * n
        + [(cfg.soc_min_kwh, cfg.soc_max_kwh)] * n
    )
    primary = np.zeros(nvar); primary[g0:g0 + n] = price
    start = perf_counter()
    first = linprog(primary, A_eq=aeq, b_eq=np.asarray(rhs), bounds=bounds, method="highs")
    if not first.success:
        raise RuntimeError(f"Q1 primary LP failed: {first.message}")
    cost_row = coo_matrix((price, (np.zeros(n, dtype=int), np.arange(n))), shape=(1, nvar)).tocsr()
    secondary = np.zeros(nvar)
    secondary[c0:c0 + n] = 1.0
    secondary[d0:d0 + n] = 1.0
    secondary[w0:w0 + n] = 1e-6
    second = linprog(
        secondary,
        A_ub=cost_row,
        b_ub=np.array([first.fun + 1e-6]),
        A_eq=aeq,
        b_eq=np.asarray(rhs),
        bounds=bounds,
        method="highs",
    )
    if not second.success:
        raise RuntimeError(f"Q1 lexicographic LP failed: {second.message}")
    elapsed = perf_counter() - start
    x = second.x
    residual = aeq @ x - np.asarray(rhs)
    return {
        "grid_kwh": x[g0:g0 + n],
        "charge_kwh": x[c0:c0 + n],
        "discharge_kwh": x[d0:d0 + n],
        "curtail_kwh": x[w0:w0 + n],
        "soc_kwh": np.r_[cfg.initial_soc_kwh, x[e0:e0 + n]],
        "cost_yuan": float(price @ x[g0:g0 + n]),
        "runtime_seconds": elapsed,
        "iterations": int((first.nit or 0) + (second.nit or 0)),
        "max_equality_residual": float(np.abs(residual).max()),
        "status": second.message,
    }


def solve_stochastic_plan(
    price_scenarios: np.ndarray,
    load_scenarios_kw: np.ndarray,
    pv_scenarios_kw: np.ndarray,
    initial_soc_kwh: float,
    cfg: ModelConfig,
    base_grid_kwh: np.ndarray | None = None,
    fixed_grid_kwh: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> PlanSolution:
    price_scenarios = np.asarray(price_scenarios, dtype=float)
    load_scenarios_kw = np.asarray(load_scenarios_kw, dtype=float)
    pv_scenarios_kw = np.asarray(pv_scenarios_kw, dtype=float)
    scenarios, n = load_scenarios_kw.shape
    if price_scenarios.shape != (scenarios, n) or pv_scenarios_kw.shape != (scenarios, n):
        raise ValueError("scenario arrays must share shape (S,n)")
    g0 = 0
    scenario0 = n
    block = 5 * n
    zeta = scenario0 + scenarios * block
    u0 = zeta + 1
    devp0 = u0 + scenarios
    devn0 = devp0 + scenarios
    up0 = devn0 + scenarios
    down0 = up0 + (n if base_grid_kwh is not None else 0)
    nvar = down0 + (n if base_grid_kwh is not None else 0)

    def idx(s: int, part: int, t: int) -> int:
        return scenario0 + s * block + part * n + t

    weights = np.full(scenarios, 1.0/scenarios) if weights is None else np.asarray(weights, float)
    if weights.shape != (scenarios,) or (weights <= 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid scenario probabilities")
    mean_price = weights @ price_scenarios
    objective = np.zeros(nvar)
    objective[g0:g0 + n] = mean_price
    for s in range(scenarios):
        objective[[idx(s, 0, t) for t in range(n)]] = cfg.emergency_price_multiplier * price_scenarios[s] * weights[s]
        objective[[idx(s, 1, t) for t in range(n)]] = cfg.throughput_epsilon * weights[s]
        objective[[idx(s, 2, t) for t in range(n)]] = cfg.throughput_epsilon * weights[s]
        terminal_unit = cfg.terminal_penalty_factor * float(np.median(price_scenarios[s])) * weights[s]
        objective[devp0 + s] = terminal_unit
        objective[devn0 + s] = terminal_unit
    objective[zeta] = cfg.cvar_weight
    objective[u0:u0 + scenarios] = cfg.cvar_weight * weights / (1.0 - cfg.cvar_alpha)
    if base_grid_kwh is not None:
        objective[up0:up0 + n] = (cfg.adjustment_up_ratio - 1) * mean_price
        objective[down0:down0 + n] = cfg.adjustment_down_ratio * mean_price

    eq_r: list[int] = []; eq_c: list[int] = []; eq_v: list[float] = []; beq: list[float] = []
    erow = 0
    for s in range(scenarios):
        for t in range(n):
            for col, value in [(g0 + t, 1.0), (idx(s, 0, t), 1.0), (idx(s, 2, t), 1.0), (idx(s, 1, t), -1.0), (idx(s, 3, t), -1.0)]:
                eq_r.append(erow); eq_c.append(col); eq_v.append(value)
            beq.append(cfg.delta_hours * (load_scenarios_kw[s, t] - pv_scenarios_kw[s, t])); erow += 1
        for t in range(n):
            for col, value in [(idx(s, 4, t), 1.0), (idx(s, 1, t), -cfg.eta_charge), (idx(s, 2, t), 1.0 / cfg.eta_discharge)]:
                eq_r.append(erow); eq_c.append(col); eq_v.append(value)
            if t:
                eq_r.append(erow); eq_c.append(idx(s, 4, t - 1)); eq_v.append(-1.0)
                beq.append(0.0)
            else:
                beq.append(initial_soc_kwh)
            erow += 1
        for col, value in [(idx(s, 4, n - 1), 1.0), (devp0 + s, -1.0), (devn0 + s, 1.0)]:
            eq_r.append(erow); eq_c.append(col); eq_v.append(value)
        beq.append(cfg.terminal_target_kwh); erow += 1
    aeq = coo_matrix((eq_v, (eq_r, eq_c)), shape=(erow, nvar)).tocsr()

    ub_r: list[int] = []; ub_c: list[int] = []; ub_v: list[float] = []; bub: list[float] = []
    urow = 0
    for s in range(scenarios):
        for t in range(n):
            ub_r.append(urow); ub_c.append(g0 + t); ub_v.append(float(price_scenarios[s, t]))
            ub_r.append(urow); ub_c.append(idx(s, 0, t)); ub_v.append(float(cfg.emergency_price_multiplier * price_scenarios[s, t]))
            if base_grid_kwh is not None:
                ub_r.append(urow); ub_c.append(up0 + t); ub_v.append(float((cfg.adjustment_up_ratio - 1) * price_scenarios[s, t]))
                ub_r.append(urow); ub_c.append(down0 + t); ub_v.append(float(cfg.adjustment_down_ratio * price_scenarios[s, t]))
        ub_r.extend([urow, urow]); ub_c.extend([zeta, u0 + s]); ub_v.extend([-1.0, -1.0]); bub.append(0.0); urow += 1
    if base_grid_kwh is not None:
        base = np.asarray(base_grid_kwh, dtype=float)
        for t in range(n):
            ub_r.extend([urow, urow]); ub_c.extend([g0 + t, up0 + t]); ub_v.extend([1.0, -1.0]); bub.append(float(base[t])); urow += 1
            ub_r.extend([urow, urow]); ub_c.extend([g0 + t, down0 + t]); ub_v.extend([-1.0, -1.0]); bub.append(float(-base[t])); urow += 1
    aub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(urow, nvar)).tocsr()

    grid_bounds = [(0.0, None)] * n if fixed_grid_kwh is None else [(float(v), float(v)) for v in fixed_grid_kwh]
    bounds = list(grid_bounds)
    for _ in range(scenarios):
        bounds += [(0.0, None)] * n
        bounds += [(0.0, cfg.interval_power_limit_kwh)] * n
        bounds += [(0.0, cfg.interval_power_limit_kwh)] * n
        bounds += [(0.0, None)] * n
        bounds += [(cfg.soc_min_kwh, cfg.soc_max_kwh)] * n
    bounds += [(None, None)] + [(0.0, None)] * scenarios + [(0.0, None)] * scenarios * 2
    if base_grid_kwh is not None:
        bounds += [(0.0, None)] * (2 * n)
    start = perf_counter()
    result = linprog(
        objective,
        A_ub=aub,
        b_ub=np.asarray(bub),
        A_eq=aeq,
        b_eq=np.asarray(beq),
        bounds=bounds,
        method="highs",
        options={"dual_feasibility_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-7},
    )
    elapsed = perf_counter() - start
    if not result.success:
        raise RuntimeError(f"stochastic LP failed: {result.message}")
    expected_soc = weights @ np.vstack([result.x[[idx(s, 4, t) for t in range(n)]] for s in range(scenarios)])
    residual = aeq @ result.x - np.asarray(beq)
    return PlanSolution(
        grid_kwh=result.x[g0:g0 + n].copy(),
        expected_soc_kwh=expected_soc,
        objective_yuan=float(result.fun),
        runtime_seconds=elapsed,
        status=result.message,
        iterations=int(result.nit or 0),
        max_equality_residual=float(np.abs(residual).max()),
        initial_soc_value=-float(sum(result.eqlin.marginals[s*(2*n+1)+n] for s in range(scenarios))),
    )
