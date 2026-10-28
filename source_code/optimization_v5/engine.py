import numpy as np

def validate(x,cfg):
    checks={}
    checks['energy_balance']=float(abs(x.actual_grid_kwh+x.emergency_kwh+x.pv_kw/6-x.pv_curtail_kwh+x.discharge_kwh-x.load_kw/6-x.charge_kwh).max())<1e-5
    checks['SOC_equation']=float(abs(x.soc_end_kwh-x.soc_start_kwh-cfg.eta_charge*x.charge_kwh+x.discharge_kwh/cfg.eta_discharge).max())<1e-5
    checks['SOC_bounds']=x.soc_end_kwh.between(cfg.soc_min_kwh-1e-5,cfg.soc_max_kwh+1e-5).all()
    checks['continuity']=np.allclose(x.soc_start_kwh.to_numpy()[1:],x.soc_end_kwh.to_numpy()[:-1],atol=1e-5,rtol=0)
    checks['PV_curtail_bound']=((x.pv_curtail_kwh>=-1e-6)&(x.pv_curtail_kwh<=x.pv_kw/6+1e-6)).all()
    checks['unused_contract_bound']=((x.unused_contract_kwh>=-1e-6)&(x.unused_contract_kwh<=x.effective_grid_kwh+1e-6)).all()
    checks['power_limit']=x[['charge_kwh','discharge_kwh']].le(cfg.interval_power_limit_kwh+1e-5).all().all()
    checks['no_simultaneous']=not ((x.charge_kwh>1e-6)&(x.discharge_kwh>1e-6)).any()
    checks['finite']=np.isfinite(x.select_dtypes('number')).all().all()
    checks['nonnegative']=x[['actual_grid_kwh','effective_grid_kwh','charge_kwh','discharge_kwh','emergency_kwh']].ge(-1e-6).all().all()
    checks['daily_slots']=x.groupby('date').size().eq(144).all()
    return {k:bool(v) for k,v in checks.items()}
