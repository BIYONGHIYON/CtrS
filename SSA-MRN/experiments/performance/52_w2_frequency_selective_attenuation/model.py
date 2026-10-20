"""Entry point to the pinned W2 C2 control and feature-frequency selective gate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'provenance/code'))
from hf_gate_model import Model,common_state,load_common,architecture_spec
