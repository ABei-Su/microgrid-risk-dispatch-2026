from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from src.scenarios import ScenarioFactory
from candidates import MatchedResidualFactory,causal_dispatch
def cvar(values,alpha=.95,weights=None):
    v=np.asarray(values,float)
    if not len(v) or not np.isfinite(v).all() or not 0<=alpha<1:raise ValueError('Invalid CVaR input')
    if weights is None:
        ordered=np.sort(v);mass=len(v)*(1-alpha);k=int(np.floor(mass));fraction=mass-k
        total=float(ordered[len(v)-k:].sum()) if k else 0.
        if fraction>1e-12:total+=fraction*ordered[len(v)-k-1]
        return total/mass
    w=np.full(len(v),1/len(v)) if weights is None else np.asarray(weights,float)
    order=np.argsort(v)[::-1];remaining=1-alpha;total=0.
    for i in order:
        take=min(float(w[i]),remaining);total+=take*v[i];remaining-=take
        if remaining<=1e-14:break
    return float(total/(1-alpha))
def cost(base,grid,emergency,price,cfg,adjusted):
    delta=grid-base
    trade=price*(base+cfg.adjustment_up_ratio*np.maximum(delta,0)-(1-cfg.adjustment_down_ratio)*np.maximum(-delta,0)) if adjusted else price*grid
    return float(np.sum(trade+cfg.emergency_price_multiplier*price*emergency))

class CachedMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.cache={};self.audit=[];self.stratified=False;self.scenario_audit=[];self.draws=[];self.last_history=[]
    def _history_dates(self,date):
        dates=super()._history_dates(date)
        if not dates or max(dates)>=date:raise ValueError('History reaches decision day')
        self.last_history=dates
        self.audit.append((str(date.date()),str(max(dates).date()),len(dates)))
        return dates
    def _residual_matrix(self,dates,column):
        key=(tuple(dates),column)
        if key not in self.cache:
            x=super()._residual_matrix(dates,column);x.setflags(write=False);self.cache[key]=x
        return self.cache[key]
    def _rng(self,date,tag):
        rng=super()._rng(date,tag)
        if not self.stratified:
            factory=self
            class RecordedRNG:
                def choice(self,values,size,replace=False):
                    result=rng.choice(values,size=size,replace=replace)
                    dates=[factory.last_history[int(i)] for i in result] if isinstance(values,(int,np.integer)) else result
                    factory.draws.append([str(pd.Timestamp(d).date()) for d in dates])
                    return result
            return RecordedRNG()

class CachedFactory(CachedMixin,ScenarioFactory):pass
class CachedMatched(CachedMixin,MatchedResidualFactory):pass

def compress_paths(arrays,k):
    # Joint standardized L1 k-medoids. Preserve stress path and probability mass.
    n=len(arrays[0])-1;k=min(k,n)
    blocks=[]
    for a in arrays:
        x=a[:-1];scale=np.std(x,axis=0);blocks.append((x-np.mean(x,axis=0))/np.maximum(scale,1e-6))
    features=np.concatenate(blocks,axis=1);dist=cdist(features,features,metric='cityblock')/features.shape[1]
    selected=[int(np.argmin(dist.sum(axis=1)))]
    while len(selected)<k:
        gains=np.min(dist[:,selected],axis=1);gains[selected]=-1;selected.append(int(np.argmax(gains)))
    for _ in range(6):
        labels=np.argmin(dist[:,selected],axis=1);updated=[]
        for j,m in enumerate(selected):
            members=np.where(labels==j)[0]
            updated.append(int(members[np.argmin(dist[np.ix_(members,members)].sum(axis=1))]) if len(members) else m)
        if updated==selected:break
        selected=updated
    labels=np.argmin(dist[:,selected],axis=1);counts=np.bincount(labels,minlength=k);keep=np.where(counts>0)[0]
    medoids=[selected[i] for i in keep];weights=np.r_[counts[keep]/n*5/6,1/6]
    return tuple(np.vstack([a[medoids],a[-1:]]) for a in arrays),weights

def scenarios(factory,date,start,official,variable,method):
    arrays=factory.official_scenarios(date,f'{start//6:02d}:00',start,include_variable_price=variable) if official else factory.standard_scenarios(date,start,include_variable_price=variable)
    if method=='cluster':
        arrays,weights=compress_paths(arrays,10)
    else:
        weights=None if method=='baseline' else np.r_[np.full(len(arrays[0])-1,(5/6)/(len(arrays[0])-1)),1/6]
    if not (official or variable) and weights is not None:
        weights[:-1]*=.96/weights[:-1].sum();weights[-1]=.04
    probabilities=np.full(len(arrays[0]),1/len(arrays[0])) if weights is None else weights
    drawn=factory.draws[-1] if factory.draws else []
    factory.scenario_audit.append(dict(date=str(pd.Timestamp(date).date()),start_slot=start,seed=factory.cfg.seed,
        method=method,scenario_count=len(arrays[0]),source_pool_count=factory.cfg.scenario_count,
        probabilities=probabilities.tolist(),sampled_historical_dates=drawn,
        unique_sampled_days=len(set(drawn)),available_history_days=len(factory.last_history),
        historical_coverage=len(set(drawn))/len(factory.last_history),
        history_last=str(max(factory.last_history).date()),stress_weight=float(probabilities[-1]),
        price_min=float(arrays[0].min()),price_max=float(arrays[0].max()),
        load_min=float(arrays[1].min()),load_max=float(arrays[1].max()),pv_min=float(arrays[2].min()),pv_max=float(arrays[2].max())))
    return arrays,weights

def evaluate(plan,arrays,state,cfg,base=None,start=0,weights=None):
    values=[];urgent=[]
    for p,l,v in zip(*arrays):
        d=causal_dispatch(plan.grid_kwh,l,v,state,cfg,plan.expected_soc_kwh,start_index=start,discharge_fraction=.35)
        values.append(cost(plan.grid_kwh if base is None else base,plan.grid_kwh,d.emergency_kwh,p,cfg,base is not None));urgent.append(d.emergency_kwh.sum())
    cv=cvar(values,cfg.cvar_alpha,weights);mean=float(np.average(values,weights=weights))
    return dict(score=mean+cfg.cvar_weight*cv,mean=mean,cvar=cv,emergency=float(np.average(urgent,weights=weights)),values=np.array(values))

def threshold(kind,old,new,os,ns,arrays,state,cfg,params):
    return cfg.update_threshold_yuan

def risk_config(cfg,actual,date,kind):
    return cfg
