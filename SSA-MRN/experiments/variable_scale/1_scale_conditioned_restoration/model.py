"""Load the preserved actual training implementation; no layer/name changes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "provenance/runtime"))
from variable_scale.model import VariableScaleNet
from variable_scale.reference import ReferenceX4
from variable_scale.engine import make_model, predict
