from pathlib import Path
import os,sys,json,hashlib,time,traceback,argparse
import runtime
from runtime import V,FROZEN,ROOT
import numpy as np
import pandas as pd
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
from src.config import ModelConfig
from replay import replay_block
from policies import cvar

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),'utf-8');os.replace(tmp,path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint():
    paths=[ROOT/'source_code/config/final_config.yaml',ROOT/'source_code/config/execution_config.json',*sorted(V.glob('*.py')),*sorted((ROOT/'source_code/src').glob('*.py')),*sorted((ROOT/'data').glob('*.csv'))]
    return hashlib.sha256(''.join(sha(p) for p in paths).encode()).hexdigest()


def transition(folder,state,**kwargs):
    path=folder/'state.json';prev=json.loads(path.read_text('utf-8')) if path.exists() else {'state':'pending'}
    allowed={'pending':{'running'},'running':{'running','completed','failed'},'failed':{'running'},'completed':{'completed'}}
    if state not in allowed[prev['state']]:raise ValueError(f'Illegal state transition {prev["state"]}->{state}')
    atomic_json(path,dict(state=state,previous=prev['state'],timestamp=time.time(),pid=os.getpid(),**kwargs))

def load_inputs():
    a=pd.read_csv(FROZEN/'data/actual_10min.csv',parse_dates=['date','interval_start'])
    p=pd.read_csv(FROZEN/'data/rolling_forecasts.csv',parse_dates=['date','interval_start'])
    o=pd.read_csv(FROZEN/'data/official_forecast_10min.csv',parse_dates=['issue_date','issue_timestamp','target_timestamp'])
    direct=pd.read_csv(FROZEN/'data/forecast_directNet.csv',parse_dates=['date','interval_start'])
    lock=json.loads((ROOT/'source_code/config/FINAL_MODEL_LOCK.json').read_text('utf-8'))
    return a,p,o,direct,{x['month']:x['selected'] for x in lock['Q3_monthly_policy']}

def run_job(job):
    params=json.loads((ROOT/'source_code/config/execution_config.json').read_text('utf-8'));signature=fingerprint();kind=job['kind'];variant=job['variant'];seed=job['seed'];smoke=job.get('smoke',False)
    ident=f'{variant}__{kind}__seed{seed}'+('__smoke' if smoke else '')
    checkpoint_area='reference_checkpoints' if job.get('fixed_reference',False) else 'checkpoints'
    folder=ROOT/'reproduced'/checkpoint_area/ident;folder.mkdir(exist_ok=True)
    manifest=folder/'identity.json'
    identity={'signature':signature,'job':job}
    if manifest.exists() and json.loads(manifest.read_text('utf-8'))!=identity:raise ValueError('Checkpoint identity mismatch: '+ident)
    if not manifest.exists():atomic_json(manifest,identity)
    if (folder/'summary.json').exists():
        summary=json.loads((folder/'summary.json').read_text('utf-8'))
        for name,h in summary['result_hashes'].items():
            if sha(folder/name)!=h:raise ValueError('Completed cache hash mismatch')
        return summary
    # Atomic exclusive lock prevents concurrent writes to the same task.
    lockpath=folder/'writer.lock'
    try:fd=os.open(lockpath,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Task writer lock exists; verify owner before resuming '+ident)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.perf_counter();transition(folder,'running',job=job)
    try:
        a,p,o,direct,choices=load_inputs();state=6000.;parts=[[],[],[]];allblocks=[]
        if job.get('fixed_reference',False):
            if variant!='q3' or kind!='baseline':raise ValueError('Reference only applies to fixed-V2 Q3')
            choices={month:'V2' for month in choices}
        months=pd.period_range('2025-02','2025-02' if smoke else '2025-12',freq='M')
        for month in months:
            pick=choices[str(month)] if variant=='q3' else 'V2';pred=direct if pick=='directNet' else p
            if variant=='q3' and pick!='directNet':pred=pd.read_csv(FROZEN/'data/forecast_V2.csv',parse_dates=['date','interval_start'])
            cfg=ModelConfig(seed=seed,cvar_weight=.02 if variant=='q2' else .06)
            if pick=='risk20':cfg=replace(cfg,scenario_count=20,cvar_weight=.12)
            if kind.startswith('joint_'):cfg=replace(cfg,scenario_count=int(kind.split('_')[1]))
            if kind=='cluster':cfg=replace(cfg,scenario_count=params['cluster_pool'])
            dates=pd.date_range(month.start_time,month.start_time if smoke else month.end_time.normalize())
            for offset in range(0,len(dates),params['checkpoint_days']):
                days=dates[offset:offset+params['checkpoint_days']];key=f'{month}_{offset:02}';meta=folder/f'{key}.json';state_in=state
                if meta.exists():
                    m=json.loads(meta.read_text('utf-8'))
                    if m['signature']!=signature or abs(m['start_soc']-state)>1e-8:raise ValueError('Cached initial state mismatch')
                    for name,h in m['hashes'].items():
                        if sha(folder/name)!=h:raise ValueError('Corrupt cached block')
                    d=pd.read_csv(folder/f'{key}_daily.csv.gz');e=pd.read_csv(folder/f'{key}_events.csv.gz');s=pd.read_csv(folder/f'{key}_solver.csv.gz')
                    state=m['end_soc']
                else:
                    x,d,e,s,audit=replay_block(a,pred,o,variant,kind,cfg,days,state,params)
                    if e.empty:e=pd.DataFrame(columns=['date','hour','kind','accepted','voi','threshold','soc'])
                    state=float(d.end_soc_kwh.iloc[-1]);hashes={}
                    for name,df in [('interval',x),('daily',d),('events',e),('solver',s)]:
                        filename=f'{key}_{name}.csv.gz';tmp=folder/(filename+'.tmp');df.to_csv(tmp,index=False,compression='gzip',float_format='%.17g');os.replace(tmp,folder/filename);hashes[filename]=sha(folder/filename)
                    m={'signature':signature,'start_soc':state_in,'end_soc':state,'days':len(days),'hashes':hashes,'history_checks':audit['history'],'scenario_checks':audit['scenarios'],'selected_prediction':pick}
                    atomic_json(meta,m)
                parts[0].append(d);parts[1].append(e);parts[2].append(s);allblocks.append(key)
                atomic_json(folder/'progress.json',{'job':ident,'last_date':str(days[-1].date()),'days_done':sum(len(df) for df in parts[0]),'end_soc':state,'elapsed':time.perf_counter()-start})
        d,e,s=[pd.concat(p,ignore_index=True) for p in parts]
        expected=1 if smoke else 334
        if len(d)!=expected or d.date.duplicated().any():raise ValueError('Incomplete full-period replay')
        if not np.allclose(d.start_soc_kwh.to_numpy()[1:],d.end_soc_kwh.to_numpy()[:-1],rtol=0,atol=1e-8):raise ValueError('Cross-day continuity failed')
        result_hashes={}
        for name,df in [('daily.csv',d),('events.csv',e),('solver.csv',s)]:df.to_csv(folder/name,index=False,float_format='%.17g');result_hashes[name]=sha(folder/name)
        summary=dict(job=ident,variant=variant,kind=kind,seed=seed,days=len(d),cost=float(d.cost_yuan.sum()),emergency=float(d.emergency_kwh.sum()),cvar95=cvar(d.cost_yuan),worst_day=float(d.cost_yuan.max()),updates=int(d.accepted_updates.sum()),end_soc=float(d.end_soc_kwh.iloc[-1]),runtime_seconds=time.perf_counter()-start,solver_seconds=float(s.seconds.sum()),solver_calls=len(s),max_lp_residual=float(s.residual.max()),complexity=params['complexity_units'][kind],checks_passed=True,smoke=smoke,signature=signature,blocks=allblocks,result_hashes=result_hashes)
        atomic_json(folder/'summary.json',summary);transition(folder,'completed',job=job);return summary
    except BaseException:
        transition(folder,'failed',error=traceback.format_exc());raise
    finally:
        lockpath.unlink(missing_ok=True)
