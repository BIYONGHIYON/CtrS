"""Entry point to the immutable original-N0 implementation shared by both arms."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'provenance/code'))
from photoaug_model import Model,common_state,load_common
