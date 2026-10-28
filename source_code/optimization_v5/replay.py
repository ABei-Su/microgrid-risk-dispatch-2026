from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import optimization_upgrade as frozen_solver
import engine as frozen_engine
from src.config import ModelConfig
from candidates import causal_dispatch
import solver
from policies import CachedFactory,CachedMatched,scenarios,evaluate,threshold,risk_config,cost

# Same solver options and arithmetic as frozen baseline where weights are unchanged.
frozen_solver.linprog=solver.linprog

def replay_block(actual,predictions,official_forecasts,variant,kind,cfg,days,initial_soc,params):
    official=variant in ('q3','q4_3');variable=variant.startswith('q4')
    factory_cls=CachedMatched if variant=='q4_2' else CachedFactory
    factory=factory_cls(actual,predictions,official_forecasts,cfg)
    factory.stratified=False
    reference=CachedFactory(actual,predictions,official_forecasts,cfg) if variant=='q4_2' else None
    method=kind if kind=='cluster' or kind.startswith('joint_') else 'baseline'
    state=float(initial_soc);intervals=[];daily=[];events=[];solvers=[]
    for date,day in actual[actual.date.isin(days)].groupby('date',sort=True):
        day=day.sort_values('interval_index');load=day.load_kw.to_numpy();pv=day.pv_actual_kw.to_numpy()
        price=day['price_actual_yuan_per_kwh' if variable else 'typical_price_yuan_per_kwh'].to_numpy()
        current=risk_config(cfg,actual,date,kind);day_start=state;updates=0
        def solve(arrays,weights,start,base=None):
            fn=frozen_solver.solve_stochastic_plan if weights is None else solver.solve_stochastic_plan
            extra={} if fn is frozen_solver.solve_stochastic_plan else {'weights':weights}
            plan=fn(*arrays,state,current,base_grid_kwh=base,**extra)
            if plan.grid_kwh.min()<-1e-6 or plan.max_equality_residual>1e-5:raise ValueError('LP feasibility failed')
            plan.grid_kwh=np.maximum(plan.grid_kwh,0.)
            solvers.append(dict(date=str(date.date()),start=start,seconds=plan.runtime_seconds,residual=plan.max_equality_residual,scenarios=len(arrays[0]),risk_weight=current.cvar_weight))
            return plan
        arrays,weights=scenarios(factory,date,0,official,variable,method);plan=solve(arrays,weights,0)
        if reference is not None:
            ref,rw=scenarios(reference,date,0,official,variable,method);old=solve(ref,rw,0)
            common=tuple(np.concatenate([x,y]) for x,y in zip(arrays,ref));cw=None if weights is None else np.r_[weights/2,rw/2]
            ns=evaluate(plan,common,state,current,weights=cw);os=evaluate(old,common,state,current,weights=cw)
            accepted=ns['score']<os['score'] and ns['emergency']<=os['emergency']+1e-6 and ns['cvar']<=os['cvar']+1e-6
            if not accepted:plan=old
            events.append(dict(date=str(date.date()),hour=0,kind='candidate_guard',accepted=accepted,voi=os['score']-ns['score'],threshold=0.,soc=state))
        base=plan.grid_kwh.copy();grid=base.copy();target=plan.expected_soc_kwh.copy()
        charge=np.zeros(144);discharge=np.zeros(144);emergency=np.zeros(144);surplus=np.zeros(144);soc=np.empty(145);soc[0]=state
        cursor=0
        for end in ([36,72,108,144] if official else [144]):
            sl=slice(cursor,end);d=causal_dispatch(grid[sl],load[sl],pv[sl],state,current,target[sl],start_index=cursor,discharge_fraction=1. if variant=='q2' else .35)
            charge[sl]=d.charge_kwh;discharge[sl]=d.discharge_kwh;emergency[sl]=d.emergency_kwh;surplus[sl]=d.curtail_kwh;soc[cursor+1:end+1]=d.soc_kwh[1:]
            state=float(d.soc_kwh[-1]);cursor=end
            if end==144:break
            arrays,weights=scenarios(factory,date,end,official,variable,method);new=solve(arrays,weights,end,base[end:])
            old=SimpleNamespace(grid_kwh=grid[end:].copy(),expected_soc_kwh=target[end:].copy())
            os=evaluate(old,arrays,state,current,base[end:],end,weights);ns=evaluate(new,arrays,state,current,base[end:],end,weights)
            gate=threshold(kind,old,new,os,ns,arrays,state,current,params);accepted=os['score']-ns['score']>gate
            prefix=grid[:end].copy()
            if accepted:grid[end:]=new.grid_kwh;target[end:]=new.expected_soc_kwh;updates+=1
            if not np.array_equal(prefix,grid[:end]):raise AssertionError('Executed prefix changed')
            events.append(dict(date=str(date.date()),hour=end//6,kind='update',accepted=accepted,voi=os['score']-ns['score'],threshold=gate,soc=state))
        curtailed=np.minimum(surplus,pv/6);unused=surplus-curtailed;times=pd.date_range(date,periods=144,freq='10min')
        x=pd.DataFrame(dict(date=date,interval_index=np.arange(144),timestamp_start=times,timestamp_end=times+pd.Timedelta(minutes=10),base_grid_kwh=base,effective_grid_kwh=grid,actual_grid_kwh=grid-unused,unused_contract_kwh=unused,pv_curtail_kwh=curtailed,curtail_kwh=surplus,charge_kwh=charge,discharge_kwh=discharge,emergency_kwh=emergency,soc_start_kwh=soc[:-1],soc_end_kwh=soc[1:],price_yuan_per_kwh=price,load_kw=load,pv_kw=pv))
        check=frozen_engine.validate(x,current)
        if not all(check.values()):raise AssertionError(check)
        total=cost(base,grid,emergency,price,current,official)
        trade=price*base if official else price*grid
        up=price*1.5*np.maximum(grid-base,0) if official else np.zeros(144)
        refund=price*.5*np.maximum(base-grid,0) if official else np.zeros(144)
        urgent=5*price*emergency
        if abs(float((trade+up-refund+urgent).sum())-total)>1e-6:raise AssertionError('Independent settlement mismatch')
        daily.append(dict(date=str(date.date()),cost_yuan=total,emergency_kwh=float(emergency.sum()),start_soc_kwh=day_start,end_soc_kwh=state,accepted_updates=updates,contract_cost=float(trade.sum()),adjustment_up_cost=float(up.sum()),refund=float(refund.sum()),emergency_cost=float(urgent.sum()),checks_passed=True))
        intervals.append(x)
    audit={'history':factory.audit+(reference.audit if reference is not None else []),'scenarios':factory.scenario_audit+(reference.scenario_audit if reference is not None else [])}
    return pd.concat(intervals,ignore_index=True),pd.DataFrame(daily),pd.DataFrame(events),pd.DataFrame(solvers),audit
