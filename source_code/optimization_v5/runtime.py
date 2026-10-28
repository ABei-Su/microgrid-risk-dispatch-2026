from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
V=Path(__file__).resolve().parent
ROOT=V.parents[1]
FROZEN=ROOT
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
sys.path[:0]=[str(V),str(ROOT/"source_code")]
