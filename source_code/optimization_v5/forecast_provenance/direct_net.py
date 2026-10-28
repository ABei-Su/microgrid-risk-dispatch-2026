import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

def build_direct_net(a,p):
    a=a.sort_values(["date","interval_index"]).reset_index(drop=True)
    p=p.sort_values(["date","interval_index"]).reset_index(drop=True)
    direct=p.copy()
    # Direct response N=L-PV, same model sees strictly lagged actuals only.
    z=pd.DataFrame(index=a.index)
    z['slot']=a.interval_index;z['dow']=a.date.dt.dayofweek;z['month']=a.date.dt.month
    for col in ['load_kw','pv_actual_kw']:
        for lag in [1,7,14,21,28]:
            z[f'{col}_lag{lag}']=a.groupby('interval_index')[col].shift(lag)
    target=a.load_kw-a.pv_actual_kw
    for month,g in p.groupby(p.date.dt.to_period('M')):
        begin=month.start_time;tr=(a.date<begin)&(a.date>=pd.Timestamp('2025-01-15'))
        te=a.date.dt.to_period('M')==month
        model=LGBMRegressor(n_estimators=90,num_leaves=15,learning_rate=.05,max_depth=-1,
                            random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(z[tr],target[tr]);n=model.predict(z[te])
        # Algebraic embedding preserves N with nonnegative L/PV.
        pv=np.maximum(g.pv_q50.to_numpy(),-n);load=n+pv
        for q in ['q10','q50','q90']:
            direct.loc[g.index,'load_'+q]=np.maximum(g['load_'+q].to_numpy()+load-g.load_q50.to_numpy(),0)
            direct.loc[g.index,'pv_'+q]=np.maximum(g['pv_'+q].to_numpy()+pv-g.pv_q50.to_numpy(),0)

    return direct
