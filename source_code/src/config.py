from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    intervals_per_day: int = 144
    delta_hours: float = 1.0 / 6.0
    capacity_kwh: float = 12000.0
    soc_min_kwh: float = 1200.0
    soc_max_kwh: float = 10800.0
    power_limit_kw: float = 5000.0
    eta_charge: float = 0.9
    eta_discharge: float = 0.9
    initial_soc_kwh: float = 6000.0
    terminal_target_kwh: float = 6000.0
    terminal_penalty_factor: float = 0.08
    emergency_price_multiplier: float = 5.0
    adjustment_down_ratio: float = 0.5
    adjustment_up_ratio: float = 1.5
    cvar_alpha: float = 0.95
    cvar_weight: float = 0.06
    scenario_count: int = 6
    residual_window_days: int = 56
    throughput_epsilon: float = 1e-7
    update_threshold_yuan: float = 1.0

    @property
    def interval_power_limit_kwh(self) -> float:
        return self.power_limit_kw * self.delta_hours
