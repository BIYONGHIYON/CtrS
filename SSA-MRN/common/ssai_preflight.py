"""Shared CPU checks for SSAI designs; no optimizer, training, or test dataset."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.model import RestoredPansharpeningNet

EXPERIMENTS = {
    1: ('1_spatial_alignment', 'SpatialAlignmentNet'),
    2: ('2_reference_area_normalization', 'ReferenceAreaNet'),
    3: ('3_calibrated_sigmoid_gating', 'CalibratedSigmoidNet'),
}


def original_logits(p, q):
    b, c, h, w = q.shape
    z = p.flatten(2) * q.transpose(-1, -2).reshape(b, c, -1)
    return z.reshape(b, c, h, w).transpose(-1, -2).reshape(b, c, -1)


def diagnose(number):
    name, cls_name = EXPERIMENTS[number]
    cls = getattr(importlib.import_module(f'experiments.ssai.{name}.model'), cls_name)
    torch.set_num_threads(1)
    records = []
    for sensor, channels in [('QB', 4), ('GF2', 4), ('WV3', 8)]:
        checkpoint = ROOT / f'research/performance/12_feature_dimension/checkpoints/runs/01_{sensor}_k4_s42/best.pt'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        torch.manual_seed(42)
        base = RestoredPansharpeningNet(channels, 4).eval()
        rng_base = torch.get_rng_state()
        torch.manual_seed(42)
        candidate = cls(channels, 4).eval()
        assert torch.equal(rng_base, torch.get_rng_state())
        assert base.state_dict().keys() == candidate.state_dict().keys()
        assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in base.state_dict().items())
        base.load_state_dict(ck['model'], strict=True)
        candidate.load_state_dict(ck['model'], strict=True)
        non_ssai = [k for k in base.state_dict() if not k.startswith(('SSA_blocks.', 'SSA_blocks1.', 'SSA_blocks2.'))]
        assert all(torch.equal(base.state_dict()[k], candidate.state_dict()[k]) for k in non_ssai)
        shapes = []
        for h, w in [(64, 64), (32, 48)]:
            torch.manual_seed(101)
            x = [torch.rand(1, 1, h, w), torch.rand(1, channels, h, w), torch.rand(1, channels, h//4, w//4)]
            with torch.inference_mode():
                a, b = base(*x), candidate(*x)
            assert b.shape == (1, channels, h, w) and torch.isfinite(b).all()
            error = float((a-b).abs().max())
            if number == 2 and (h, w) == (64, 64):
                assert torch.equal(a, b)
            shapes.append({'pan_hw': [h, w], 'output_shape': list(b.shape), 'finite': True,
                           'baseline_max_absolute_difference': error})
        torch.manual_seed(102)
        x = [torch.rand(1, 1, 64, 64), torch.rand(1, channels, 64, 64), torch.rand(1, channels, 16, 16)]
        candidate.train()
        candidate(*x).square().mean().backward()
        grads = {k: p.grad for k, p in candidate.named_parameters()}
        assert all(torch.isfinite(v).all() for v in grads.values() if v is not None)
        gradient_equal = None
        if number == 2:
            base.train()
            base(*x).square().mean().backward()
            gradient_equal = all((p.grad is None and grads[k] is None) or
                                 (p.grad is not None and grads[k] is not None and torch.equal(p.grad, grads[k]))
                                 for k, p in base.named_parameters())
            assert gradient_equal
        candidate.eval()
        macs = [0]
        def count(module, inputs, output):
            macs[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        hooks = [m.register_forward_hook(count) for m in candidate.modules() if isinstance(m, torch.nn.Conv2d)]
        with torch.inference_mode(): candidate(*x)
        for hook in hooks: hook.remove()
        records.append({'sensor': sensor, 'checkpoint': str(checkpoint.relative_to(ROOT)),
                        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        'epoch': ck['epoch'], 'strict_load': True, 'same_initial_state_and_rng': True,
                        'non_ssai_state_unchanged': True, 'parameters': sum(p.numel() for p in candidate.parameters()),
                        'finite_gradients_without_optimizer_step': True, 'gradient_equal_to_baseline_at_64': gradient_equal,
                        'unused_parameters_without_gradient': sum(p.numel() for p in candidate.parameters() if p.grad is None),
                        'executed_conv_macs_64': macs[0], 'shapes': shapes})
    torch.manual_seed(42)
    p, q = [torch.randn(1, 4, 16, 16, dtype=torch.float64) for _ in range(2)]
    z = original_logits(p, q)
    original = lambda p, q: p * original_logits(p, q).softmax(-1).reshape_as(p)
    aligned = lambda p, q: p * (p*q).flatten(2).softmax(-1).reshape_as(p)
    shift = lambda v: torch.roll(v, shifts=(1, 0), dims=(-2, -1))
    formula_error = float((z.reshape_as(p) - p.transpose(-1, -2)*q).abs().max())
    assert formula_error == 0
    checks = {'square_original_formula_error': formula_error,
              'feature_periodic_shift_original_max_error': float((original(shift(p), shift(q))-shift(original(p,q))).abs().max()),
              'feature_periodic_shift_aligned_max_error': float((aligned(shift(p), shift(q))-shift(aligned(p,q))).abs().max()),
              'scope': 'Projected synthetic features only; not whole-model translation equivariance or performance.'}
    if number == 3:
        gate = (2/4096) * z.sigmoid()
        checks['calibrated_gate_min'] = float(gate.min())
        checks['calibrated_gate_max'] = float(gate.max())
        checks['gate_upper_bound'] = 2/4096
        assert bool(((gate >= 0) & (gate <= 2/4096)).all())
    sources = [Path(__file__), ROOT/'common/model.py', ROOT/'references/upstream/network.py',
               ROOT/f'experiments/ssai/{name}/model.py']
    return {'experiment': number, 'kind': 'synthetic_preflight_not_performance', 'torch_version': torch.__version__,
            'device': 'cpu', 'precision': 'float32_model_float64_formula', 'training_started': False,
            'optimizer_steps': 0, 'dataset_used': False, 'records': records, 'analytical_checks': checks,
            'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'limits': 'No real-data validation/RR/FR, GPU throughput or peak memory measurements. Conv MACs exclude interpolation, softmax/sigmoid and memory traffic.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=int, choices=EXPERIMENTS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = diagnose(args.experiment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'experiment': args.experiment, 'passed': True, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
