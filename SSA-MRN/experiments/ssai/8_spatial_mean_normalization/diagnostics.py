"""Read-only, RNG-preserving diagnostics shared by preflight and epoch reports."""
import contextlib, hashlib, math, random
import numpy as np
import torch
from torch import nn

STAGES = ('SSA_blocks', 'SSA_blocks1', 'SSA_blocks2')
FUSIONS = ('conv48tnum1', 'conv48tnum2', 'conv48tnum3')


@contextlib.contextmanager
def preserve_rng():
    py, npstate = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try:
            yield
        finally:
            random.setstate(py); np.random.set_state(npstate)


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def weight_changes(model, initial):
    # Convolution weights only: omit bias and, critically, gamma.
    state = model.state_dict()
    def measure(prefix):
        keys = [k for k in initial if k.startswith(prefix + '.') and k.endswith('.weight') and initial[k].ndim == 4 and '.conv6t12.' not in k]
        numerator = sum((state[k].detach().cpu().double()-initial[k].double()).square().sum().item() for k in keys)
        denominator = sum(initial[k].double().square().sum().item() for k in keys)
        return math.sqrt(numerator / denominator) if denominator else None
    names = list(STAGES) + list(FUSIONS) + [f'{stage}.{band}' for stage in STAGES for band in range(4)]
    return {name: measure(name) for name in names}


def rms_ratios(model, batches, device):
    """Ratios of globally pooled RMS; per-band values and stage aggregate, no averaging of ratios."""
    sums = {}
    handles = []
    training = model.training
    def record(key, out, ref):
        target = sums.setdefault(key, [0., 0., 0, 0])
        target[0] += out.detach().double().square().sum().item()
        target[1] += ref.detach().double().square().sum().item()
        target[2] += out.numel(); target[3] += ref.numel()
    for i, stage in enumerate(STAGES):
        for band, block in enumerate(getattr(model, stage)):
            def hook(module, args, output, name=f'{stage}.{band}'):
                pan, band_input = args
                p = module.conv6t6(module.relu(module.conv1t6(pan)))
                record(name, output, p)
            handles.append(block.register_forward_hook(hook))
        def fusion_hook(module, args, output, index=i, stage=stage):
            # The first block sees the stage's residual before fusion is added.
            module._diagnostic_res = args[1][:, :4].detach()
        first = getattr(model, stage)[0]
        handles.append(first.register_forward_hook(fusion_hook))
        def activation_hook(module, args, output, index=i, first=first):
            record(FUSIONS[index], output, first._diagnostic_res)
            del first._diagnostic_res
        handles.append(getattr(model, f'prelu{i+1}').register_forward_hook(activation_hook))
    try:
        with preserve_rng(), torch.inference_mode():
            model.eval()
            for batch in batches:
                model(*(x.to(device) for x in batch[:3]))
    finally:
        for handle in handles: handle.remove()
        model.train(training)
    def ratio(values):
        a, b, na, nb = values
        return math.sqrt((a / na) / (b / nb)) if b else None
    result = {'block_output_over_p': {}, 'pan_ms_over_res': {}, 'aggregation': 'sqrt(pooled output mean square / pooled reference mean square); output includes gamma'}
    for stage in STAGES:
        values = [sums[f'{stage}.{band}'] for band in range(4)]
        result['block_output_over_p'][stage] = {'all_bands': ratio([sum(v[j] for v in values) for j in range(4)]), 'bands': [ratio(v) for v in values]}
    result['pan_ms_over_res'] = {name: ratio(sums[name]) for name in FUSIONS}
    return result


def gradient_groups(model, batches, device):
    groups = {'backbone': [], 'conv64t48': [], **{s: [] for s in STAGES}, **{s: [] for s in FUSIONS}, 'decoder': [], 'other': []}
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    for name, p in model.named_parameters():
        prefix = name.split('.')[0]
        group = 'backbone' if id(p) in backbone_ids else (prefix if prefix in groups else ('decoder' if prefix.startswith('conv2ctc') else 'other'))
        groups[group].append(p)
    rows = {name: [] for name in groups}
    active_counts = {}
    count = 0
    with preserve_rng():
        model.train()
        for batch in batches:
            model.zero_grad(set_to_none=True)
            pan, lms, ms, gt = [x.to(device) for x in batch]
            nn.functional.mse_loss(model(pan, lms, ms), gt).backward()
            for name, params in groups.items():
                rows[name].append(math.sqrt(sum(p.grad.detach().double().square().sum().item() for p in params if p.grad is not None)))
                active_counts[name] = sum(p.numel() for p in params if p.grad is not None)
            count += 1
    model.zero_grad(set_to_none=True)
    return {'batches': count, 'optimizer_updates': 0, 'groups': {name: {'parameter_count': sum(p.numel() for p in groups[name]), 'parameters_with_gradient': active_counts[name], 'mean_gradient_l2': float(np.mean(values)), 'mean_gradient_rms': float(np.mean(values))/math.sqrt(active_counts[name]) if active_counts[name] else None} for name, values in rows.items()}}
