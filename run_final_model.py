"""Launch the submission's portable code entry."""
from pathlib import Path
import runpy,sys
entry=Path(__file__).resolve().parent/'source_code/optimization_v5/run_final_model.py'
sys.path.insert(0,str(entry.parent))
runpy.run_path(str(entry),run_name='__main__')
