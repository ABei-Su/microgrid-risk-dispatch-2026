"""Named question entry points; delegate to unchanged v5 solvers."""
import runtime
from runtime import ROOT
import run_experiments as runner
from solver import solve_q1 as Q1_solver

def _run(variant,kind,seed):
    (ROOT/'reproduced/checkpoints').mkdir(parents=True,exist_ok=True)
    return runner.run_job(dict(variant=variant,kind=kind,seed=seed,smoke=False))

def Q2_CVaR_optimizer(seed=1):return _run('q2','joint_50',seed)
def Q3_MPC_controller(seed=1):return _run('q3','baseline',seed)
def Q4_day_ahead_optimizer(seed=1):return _run('q4_2','cluster',seed)
def Q4_intraday_optimizer(seed=1):return _run('q4_3','cluster',seed)
