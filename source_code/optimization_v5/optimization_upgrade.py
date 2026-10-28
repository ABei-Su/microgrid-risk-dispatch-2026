from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,vstack
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

def solve_stochastic_plan(
    price_scenarios: np.ndarray,
    load_scenarios_kw: np.ndarray,
    pv_scenarios_kw: np.ndarray,
    initial_soc_kwh: float,
    cfg: ModelConfig,
    base_grid_kwh: np.ndarray | None = None,
    fixed_grid_kwh: np.ndarray | None = None,
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

    mean_price = price_scenarios.mean(axis=0)
    objective = np.zeros(nvar)
    objective[g0:g0 + n] = mean_price
    for s in range(scenarios):
        objective[[idx(s, 0, t) for t in range(n)]] = cfg.emergency_price_multiplier * price_scenarios[s] / scenarios
        objective[[idx(s, 1, t) for t in range(n)]] = cfg.throughput_epsilon / scenarios
        objective[[idx(s, 2, t) for t in range(n)]] = cfg.throughput_epsilon / scenarios
        terminal_unit = cfg.terminal_penalty_factor * float(np.median(price_scenarios[s])) / scenarios
        objective[devp0 + s] = terminal_unit
        objective[devn0 + s] = terminal_unit
    objective[zeta] = cfg.cvar_weight
    objective[u0:u0 + scenarios] = cfg.cvar_weight / ((1.0 - cfg.cvar_alpha) * scenarios)
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
    expected_soc = np.mean(
        np.vstack([result.x[[idx(s, 4, t) for t in range(n)]] for s in range(scenarios)]), axis=0
    )
    residual = aeq @ result.x - np.asarray(beq)
    return PlanSolution(
        grid_kwh=result.x[g0:g0 + n].copy(),
        expected_soc_kwh=expected_soc,
        objective_yuan=float(result.fun),
        runtime_seconds=elapsed,
        status=result.message,
        iterations=int(result.nit or 0),
        max_equality_residual=float(np.abs(residual).max()),
    )
