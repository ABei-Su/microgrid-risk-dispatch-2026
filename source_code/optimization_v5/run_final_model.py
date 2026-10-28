"""Portable v5 entry: frozen forecasts, unchanged solvers, new output directory."""
import argparse,json,hashlib
import runtime
from runtime import ROOT,V
import numpy as np
import pandas as pd
import run_experiments as runner
from src.config import ModelConfig
from solver import solve_q1
from policies import cvar

def configuration():
    cfg=json.loads((ROOT/'source_code/config/final_config.yaml').read_text('utf-8'))
    lock=json.loads((ROOT/'source_code/config/FINAL_MODEL_LOCK.json').read_text('utf-8'))
    assert cfg['status']=='adopted' and cfg['seeds']==[1,2,3,4,5]
    assert cfg['physical_and_base_parameters']==lock['physical_and_default_parameters']
    assert cfg['models']['q2']['kind']=='joint_50'
    assert cfg['models']['q3']['kind']=='baseline'
    assert cfg['models']['q3']['monthly_policy']==lock['Q3_monthly_policy']
    assert cfg['models']['q4_2']['kind']==cfg['models']['q4_3']['kind']=='cluster'
    manifest=json.loads((ROOT/'source_code/config/package_integrity.json').read_text('utf-8'))
    for name,digest in manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Package integrity mismatch: '+name)
    return cfg

def q1(out):
    day=pd.read_csv(ROOT/'data/typical_day.csv');cfg=ModelConfig(seed=1)
    r=solve_q1(day.typical_price_yuan_per_kwh.to_numpy(),day.typical_load_kw.to_numpy(),day.typical_pv_kw.to_numpy(),cfg)
    baseline=float(np.maximum((day.typical_load_kw-day.typical_pv_kw).to_numpy()/6,0)@day.typical_price_yuan_per_kwh.to_numpy())
    assert abs(r['cost_yuan']-35126.94859)<1e-4
    x=pd.DataFrame(dict(slot=range(144),grid_kwh=r['grid_kwh'],charge_kwh=r['charge_kwh'],discharge_kwh=r['discharge_kwh'],soc_start_kwh=r['soc_kwh'][:-1],soc_end_kwh=r['soc_kwh'][1:]))
    x.to_csv(out/'Q1_trajectory.csv',index=False,float_format='%.17g')
    d=dict(cost_yuan=r['cost_yuan'],grid_kwh=float(r['grid_kwh'].sum()),no_storage_cost_yuan=baseline,savings_pct=100*(baseline-r['cost_yuan'])/baseline)
    pd.DataFrame([d]).to_csv(out/'Q1_summary.csv',index=False,float_format='%.17g');return d

def verify_results(out):
    """Recompute final statistics from delivered daily CSVs, without optimization."""
    records=[]
    for variant,folder in [('q2','Q2'),('q3','Q3'),('q4_2','Q4'),('q4_3','Q4')]:
        for seed in range(1,6):
            d=pd.read_csv(ROOT/f'result/{folder}/{variant}_seed{seed}/daily.csv')
            assert len(d)==334 and not d.date.duplicated().any()
            assert np.allclose(d.start_soc_kwh.to_numpy()[1:],d.end_soc_kwh.to_numpy()[:-1],atol=1e-7,rtol=0)
            assert np.allclose(d.contract_cost+d.adjustment_up_cost-d.refund+d.emergency_cost,d.cost_yuan,atol=1e-6,rtol=0)
            records.append(dict(variant=variant,seed=seed,cost=d.cost_yuan.sum(),emergency=d.emergency_kwh.sum(),cvar95=cvar(d.cost_yuan),updates=d.accepted_updates.sum()))
    r=pd.DataFrame(records);r.to_csv(out/'five_seed_recomputed.csv',index=False,float_format='%.17g')
    expected=pd.read_csv(ROOT/'validation/final_results.csv')
    for row in records:
        e=expected[(expected.variant==row['variant'])&(expected.seed==row['seed'])].iloc[0]
        for key in ['cost','emergency','cvar95','updates']:
            if not np.isclose(row[key],e[key],atol=1e-6,rtol=0):raise AssertionError((row['variant'],row['seed'],key))
    stats=r.groupby('variant')[['cost','emergency','cvar95','updates']].agg(['mean','std','min','max']);stats.to_csv(out/'five_seed_statistics.csv',float_format='%.17g')
    daily=pd.read_csv(ROOT/'result/Q3/Q3_paired_daily.csv')
    for seed in range(1,6):
        old=pd.read_csv(ROOT/f'result/Q3/fixed_reference_seed{seed}/daily.csv').set_index('date')
        new=pd.read_csv(ROOT/f'result/Q3/q3_seed{seed}/daily.csv').set_index('date')
        paired=daily[daily.seed==seed].set_index('date').loc[new.index]
        assert old.index.equals(new.index)
        assert np.allclose(old.cost_yuan-new.cost_yuan,paired.gain,rtol=0,atol=1e-8)
    matrix=daily.pivot(index='date',columns='seed',values='gain').to_numpy();rng=np.random.default_rng(20260912);draws=[]
    for _ in range(10000):
        starts=rng.integers(0,len(matrix)-7+1,size=int(np.ceil(len(matrix)/7)))
        idx=np.concatenate([np.arange(s,s+7) for s in starts])[:len(matrix)]
        draws.append(matrix[idx].sum(axis=0))
    ci=np.quantile(np.asarray(draws).mean(axis=1),[.025,.975]);ref=json.loads((ROOT/'result/Q3/Q3_bootstrap.json').read_text('utf-8'))
    assert np.allclose(ci,ref['mean_gain_ci95'],rtol=0,atol=1e-6)
    (out/'verification.json').write_text(json.dumps(dict(passed=True,annual_rows=20,days_per_seed=334,bootstrap_ci95=ci.tolist()),indent=2),'utf-8')
    print(stats.to_string(),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--question',choices=['all','q1','q2','q3','q4_2','q4_3'],default='all')
    ap.add_argument('--seed',type=int,choices=range(1,6))
    ap.add_argument('--verify-only',action='store_true',help='Recompute archived statistics and Q1; no annual re-solve')
    ap.add_argument('--smoke',action='store_true',help='One-day entry check; never a full annual result')
    ap.add_argument('--with-reference',action='store_true',help='Also solve the delivered Q3 fixed-V2 paired reference')
    args=ap.parse_args();cfg=configuration()
    out=ROOT/'reproduced';out.mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True);(out/'reference_checkpoints').mkdir(exist_ok=True)
    if args.verify_only:
        print('Q1',q1(out));verify_results(out);return
    questions=['q1','q2','q3','q4_2','q4_3'] if args.question=='all' else [args.question]
    if 'q1' in questions:print('Q1',q1(out),flush=True)
    seeds=[args.seed] if args.seed else ([1] if args.smoke else cfg['seeds'])
    results=[]
    for question in questions:
        if question=='q1':continue
        for seed in seeds:
            print('RUN',question,seed,'one day' if args.smoke else '334 days',flush=True)
            r=runner.run_job(dict(variant=question,kind=cfg['models'][question]['kind'],seed=seed,smoke=args.smoke));results.append(r)
            print('DONE',r['job'],r['days'],r['cost'],flush=True)
    if results:
        frame=pd.DataFrame(results)
        frame.to_csv(out/('entry_smoke.csv' if args.smoke else 'final_replay.csv'),index=False)
        if not args.smoke:frame.groupby('variant')[['cost','emergency','cvar95','updates']].agg(['mean','std','min','max']).to_csv(out/'requested_replay_statistics.csv')
    if not args.smoke:
        verify_results(out)
        expected=pd.read_csv(ROOT/'validation/final_results.csv')
        for r in results:
            e=expected[(expected.variant==r['variant'])&(expected.seed==r['seed'])].iloc[0]
            for key in ['cost','emergency','cvar95','updates']:
                if not np.isclose(r[key],e[key],rtol=0,atol=1e-4):raise AssertionError(('Replay differs',r['job'],key,r[key],e[key]))
        print('Full requested replay agrees with delivered final results.',flush=True)
    if args.with_reference and 'q3' in questions:
        references=[]
        for seed in seeds:
            r=runner.run_job(dict(variant='q3',kind='baseline',seed=seed,smoke=args.smoke,fixed_reference=True))
            references.append(r)
            if not args.smoke:
                old=pd.read_csv(ROOT/f'result/Q3/fixed_reference_seed{seed}/daily.csv')
                assert np.isclose(r['cost'],old.cost_yuan.sum(),rtol=0,atol=1e-4)
            print('REFERENCE DONE',seed,r['days'],r['cost'],flush=True)
        pd.DataFrame(references).to_csv(out/'Q3_reference_replay.csv',index=False)

if __name__=='__main__':main()
