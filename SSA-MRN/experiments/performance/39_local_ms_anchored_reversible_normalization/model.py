"""Entry point to the immutable global-N1/local normalization implementation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'provenance/code'))
from localnorm_model import Model,common_state,load_common
