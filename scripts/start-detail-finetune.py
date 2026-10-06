"""Start ten weights-only fine-tuning epochs after a completed full detail run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.from_dir.resolve() / 'best.pt'
    state = torch.load(source, map_location='cpu', weights_only=True)
    cfg = state['config']
    if (cfg.get('model_type') != 'rgb_pilot_joint17_23tap' or
            cfg.get('pilot_options') != {'detail_path': True} or
            cfg.get('pilot_train_scenes') is not None):
        parser.error('Expected a full-scene detail checkpoint')
    history = [json.loads(x) for x in (source.parent/'history.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if not history or history[-1]['epoch'] != 100:
        parser.error('Complete the 100-epoch run before fine-tuning')
    config = dict(cfg)
    config.update(epochs=10, learning_rate=1e-5, init_checkpoint=str(source))
    output = root / 'SSA-MRN/experiments/logs/remote-control-detail' / ('finetune-' + source.parent.name + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, str(root/'scripts/remote-training-detail.py'), 'start', '--config', str(output)], check=True)


if __name__ == '__main__':
    main()
