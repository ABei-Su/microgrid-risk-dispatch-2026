from dataclasses import dataclass
import numpy as np

@dataclass
class DispatchResult:
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    emergency_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_kwh: np.ndarray
