"""Load the immutable trained W2/gated-W2 implementation without rewriting its layers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'provenance/code'))
from gate29_model import W2Net, common_state, load_common
