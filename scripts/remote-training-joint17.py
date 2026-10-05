"""Dedicated disconnect-safe controller for the joint-17 LIB-HSI experiment."""
import importlib.util
from pathlib import Path

script = Path(__file__).with_name('remote-training.py')
spec = importlib.util.spec_from_file_location('ctrs_remote_training', script)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
controller.BASE = controller.ROOT / 'SSA-MRN/experiments/logs/remote-control-joint17'
controller.RUNS = controller.ROOT / 'SSA-MRN/experiments/checkpoints/remote-runs-joint17'
controller.CONFIG = controller.ROOT / 'SSA-MRN/configs/lib_rgb_hsi_joint17_k4_consistency_tiles.json'

if __name__ == '__main__':
    controller.main()
