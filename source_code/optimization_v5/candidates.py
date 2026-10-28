import numpy as np
from src.config import ModelConfig
from src.scenarios import ScenarioFactory
from src.simulator import DispatchResult

def causal_dispatch(grid_kwh,load_kw,pv_kw,initial_soc_kwh,cfg,target_soc_kwh=None,*,start_index=0,discharge_fraction=.35):
    """A known-prefix-invariant policy; day-clock, not array length, controls reserve."""
    arrays=[np.asarray(x,dtype=float) for x in (grid_kwh,load_kw,pv_kw)]
    if any(a.ndim!=1 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Inputs must be finite 1D arrays')
    grid,load,pv=arrays; n=len(grid)
    if len(load)!=n or len(pv)!=n or start_index<0 or start_index+n>144:
        raise ValueError('Shape/day-window mismatch')
    if any((a<0).any() for a in arrays): raise ValueError('Negative physical input')
    if not cfg.soc_min_kwh<=initial_soc_kwh<=cfg.soc_max_kwh:
        raise ValueError('Initial SOC outside bounds')
    target=np.full(n,cfg.terminal_target_kwh) if target_soc_kwh is None else np.asarray(target_soc_kwh,dtype=float)
    if target.shape!=(n,) or not np.isfinite(target).all(): raise ValueError('Invalid target SOC')
    charge=np.zeros(n); discharge=np.zeros(n); emergency=np.zeros(n); surplus=np.zeros(n)
    soc=np.empty(n+1); soc[0]=initial_soc_kwh
    for t in range(n):
        deficit=cfg.delta_hours*(load[t]-pv[t])-grid[t]
        if deficit>0:
            physical=min(cfg.interval_power_limit_kwh,max(0.,cfg.eta_discharge*(soc[t]-cfg.soc_min_kwh)))
            floor=max(cfg.soc_min_kwh,min(cfg.soc_max_kwh,target[t])-1200.)
            preferred=max(0.,cfg.eta_discharge*(soc[t]-floor))
            allowed=physical if start_index+t>=126 else min(physical,max(preferred,discharge_fraction*physical))
            discharge[t]=min(deficit,allowed); emergency[t]=deficit-discharge[t]
        else:
            charge[t]=min(-deficit,cfg.interval_power_limit_kwh,max(0.,(cfg.soc_max_kwh-soc[t])/cfg.eta_charge))
            surplus[t]=-deficit-charge[t]
        soc[t+1]=soc[t]+cfg.eta_charge*charge[t]-discharge[t]/cfg.eta_discharge
    # `curtail_kwh` retained only for compatibility: represents total unused supply,
    # not pure PV curtailment. Experimental output names this distinction explicitly.
    return DispatchResult(charge,discharge,emergency,surplus,soc)

class MatchedResidualFactory(ScenarioFactory):
    def _residual_matrix(self,dates,column):
        name={'load_kw':'load','pv_actual_kw':'pv','price_actual_yuan_per_kwh':'price'}[column]
        rows=[]
        for d in dates:
            actual=self.actual_by_date[d]
            if d in self.pred_by_date:
                prediction=self.pred_by_date[d][f'{name}_q50'].to_numpy(float)
            else:
                prediction=actual[f'{column}_lag7d'].to_numpy(float)
                if not np.isfinite(prediction).all(): raise ValueError('Missing historical prediction')
            rows.append(actual[column].to_numpy(float)-prediction)
        values=np.vstack(rows)
        return values-np.median(values,axis=0,keepdims=True)
