"""Experiment 68 (no training): three diagnostics on archived weights, local CPU. Rules are fixed in config.json.

D1  C2MS SSAI folding: is the SSA attention of C2MS dead as in A0 (SSAI 11)? Variants A, PRUNED (upstream layers that
    forward never calls deleted), Z_SSA (SSA outputs zero), FOLDED (standalone module, each SSAI stage = its per-band
    constant PReLU_k(bias_k)), Z_STAGE (stage outputs zero). Also SSA output sizes, parameter counts and latency.
D2  Frequency split of the 65 SAM gain: Shapley split of S(C2MS) - S(65) into the low band L = MS-MTF blur (60) and the
    high band H = y - L(y), the H part further into per-pixel magnitude |H| and band ratio H/|H| (61), and the remaining
    LF headroom of 65 (60 LF_fixed oracle).
D3  Seed ensembles E_C2MS, E_L1S (mean of seeds 42-44) and E_ALL (six outputs) on QB validation (decision) and on the
    RR/RR2/FR_WALD/FR tests of QB, WV3 and WV2 (transfer), with the same LF/HF split against each member.

Parts: --part check (FFT operators vs the direct 54 form), test (per target, writes results/raw_<target>.json), val (QB validation), latency, images, summary
(aggregates, gates, results/summary.json and diagnostics.png). Writes only below results/.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse, copy, hashlib, importlib.util, json, sys, time, types
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]; PERF = ROOT / 'experiments/performance'
sys.path.insert(0, str(ROOT))
from common.metrics import rr_metrics  # noqa: E402  (imported before the 43 provenance path is added)
from common import result_images  # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


E54 = PERF / '54_multi_sensor_scale_mixed_training'
M54 = _module('exp54_model', E54 / 'model.py')  # 54 Model('C2MS'): 53 C2MS for any band count (65 uses the same)
P54 = _module('exp54_prepare', E54 / 'prepare.py')
SENSORS, Wald, gen_mtf, _blur = P54.SENSORS, P54.Wald, P54.gen_mtf, P54._blur
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
SEEDS = CFG['seeds']; OUT = HERE / 'results'
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS'])); torch.set_grad_enabled(False)

# Direct children of RestoredPansharpeningNet that forward() calls (with parameters); every other parametrized child is unused.
USED = ('conv1t1', 'prelu', 'cov2t64', 'backbone', 'conv64t48', 'SSA_blocks', 'SSA_blocks1', 'SSA_blocks2',
        'conv48tnum1', 'conv48tnum2', 'conv48tnum3', 'prelu1', 'prelu2', 'prelu3', 'conv2ctc1', 'conv2ctc2', 'conv2ctc3')
STAGES = (('SSA_blocks', 'conv48tnum1', 'prelu1'), ('SSA_blocks1', 'conv48tnum2', 'prelu2'), ('SSA_blocks2', 'conv48tnum3', 'prelu3'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def count(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------- models

def load_net(family, sensor, seed):
    path = ROOT / CFG['checkpoints'][family][sensor].replace('<seed>', str(seed))
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = M54.Model('C2MS', SENSORS[sensor]['bands'], seed); net.load_state_dict(ck['model'], strict=True); net.eval()
    info = dict(checkpoint=str(path.relative_to(ROOT)), sha256=sha(path), best_epoch=ck.get('best_epoch'), archived_val=ck.get('metrics'))
    return net, info


_ATTEND = None


def add_modes(net):
    """Install a switchable _attend on the core: mode A (original), Z_SSA (SSA outputs zero), Z_STAGE (stage output zero)."""
    global _ATTEND
    core = net.core; _ATTEND = _ATTEND or type(core)._attend

    def attend(self, pan, features, ms, blocks, fusion, activation):
        mode = self.diag_mode
        if mode == 'A': return _ATTEND(self, pan, features, ms, blocks, fusion, activation)
        n, _, h, w = features.shape
        if mode == 'Z_SSA': return activation(fusion(features.new_zeros(n, fusion.in_channels, h, w)))
        if mode == 'Z_STAGE': return features.new_zeros(n, self.channels, h, w)
        raise ValueError(mode)
    core._attend = types.MethodType(attend, core); core.diag_mode = 'A'
    return net


def pruned(net):
    """Copy of the network with every parametrized child that forward never calls deleted (and SSA conv6t12)."""
    p = copy.deepcopy(net); core = p.core
    used = {id(t) for name in USED for t in getattr(core, name).parameters()}
    removed = {}
    for name, child in list(core.named_children()):
        params = list(child.parameters())
        if params and not any(id(t) in used for t in params):
            removed[name] = sum(t.numel() for t in params); delattr(core, name)
    ssa_extra = 0
    for blocks, _, _ in STAGES:
        for block in getattr(core, blocks):
            ssa_extra += count(block.conv6t12); del block.conv6t12
    removed['SSA.conv6t12 (all blocks)'] = ssa_extra
    core.diag_mode = 'A'
    return p, removed


class Folded(nn.Module):
    """C2MS without the SSA blocks and the band fusion convs. With the SSA outputs at zero, stage k of SSAI returns the
    per-band constant PReLU_k(bias of conv48tnum_k) everywhere, so: res += c1, res_half = down(res) + c2, and the
    quarter-resolution input of the decoder is the constant image c3 (equal to Z_SSA by construction)."""

    def __init__(self, net):
        super().__init__(); c = net.core
        for name in ('conv1t1', 'prelu', 'cov2t64', 'backbone', 'conv64t48', 'conv2ctc1', 'conv2ctc2', 'conv2ctc3',
                     'downsample1', 'upsample1', 'downsample100', 'upsample101', 'upsample102'):
            setattr(self, name, copy.deepcopy(getattr(c, name)))
        for k, (_, conv, act) in enumerate(STAGES, 1):
            self.register_buffer(f'const{k}', getattr(c, act)(getattr(c, conv).bias.detach().clone()[None, :, None, None]))

    def forward(self, pan, lms, ms):
        pan_updown = self.prelu(self.conv1t1(self.upsample1(self.downsample1(pan))))
        res = self.conv64t48(self.backbone(self.cov2t64(torch.cat([lms, pan, pan_updown], dim=1)))) + lms
        res = res + self.const1
        res_half = self.downsample100(res) + self.const2
        output = self.conv2ctc1(torch.cat([ms, self.const3.expand(ms.shape)], dim=1))
        output = self.conv2ctc2(torch.cat([self.upsample101(output), res_half], dim=1))
        return self.conv2ctc3(torch.cat([self.upsample102(output), res], dim=1))


class Ensemble(nn.Module):
    def __init__(self, nets):
        super().__init__(); self.nets = nn.ModuleList(nets)

    def forward(self, pan, lms, ms):
        return torch.stack([n(pan, lms, ms) for n in self.nets]).mean(0)


def param_table(net):
    core = net.core; p, removed = pruned(net); f = Folded(net)
    ssa = sum(count(b) - count(b.conv6t12) for blocks, _, _ in STAGES for b in getattr(core, blocks))
    fusion = sum(count(getattr(core, conv)) + count(getattr(core, act)) for _, conv, act in STAGES)
    return dict(A=count(net), unused_removed=sum(removed.values()), unused_by_module=removed, PRUNED=count(p),
                ssa_blocks_used=ssa, band_fusion_and_prelu=fusion, FOLDED=count(f), FOLDED_constants=sum(b.numel() for b in f.buffers()),
                w2_branch=count(core.backbone.branch))


# ---------------------------------------------------------------- data and metrics

def scenes(target, kind):
    tag = SENSORS[target]['tag']; name = f'test_{tag}_multiExm1.h5' if kind == 'RR' else f'test_{tag}_OrigScale_multiExm1.h5'
    path = ROOT / 'data/test' / name
    if sha(path) != CFG['data_sha256'][target][kind]: raise ValueError(f'data sha mismatch {path}')
    with h5py.File(path, 'r') as f:
        keys = [k for k in ('pan', 'lms', 'ms', 'gt') if k in f]
        return [{k: np.asarray(f[k][i], dtype=np.float64) for k in keys} for i in range(len(f['pan']))]


def predict(net, x, scale):
    t = [torch.from_numpy((np.asarray(a, dtype=np.float64) / scale).astype('float32')).unsqueeze(0) for a in x]
    y = net(*t)[0].float().numpy().astype(np.float64) * scale
    if not np.isfinite(y).all(): raise FloatingPointError('nonfinite prediction')
    return y


def sam_psnr(g, y):
    """SAM and PSNR exactly as common/metrics.rr_metrics (checked against it on the A outputs)."""
    dot = np.sum(g * y, axis=0); norms = np.linalg.norm(g, axis=0) * np.linalg.norm(y, axis=0); valid = norms > 0
    sam = float(np.degrees(np.arccos(np.clip(dot[valid] / norms[valid], -1, 1))).mean())
    band_mse = np.mean((y - g) ** 2, axis=(1, 2))
    return sam, float(np.mean(10 * np.log10(2047 * 2047 / np.maximum(band_mse, 1e-30))))


def fft_blur(x, kernel):
    """54 prepare._blur (replicate border, depthwise 41x41 cross-correlation) computed by FFT as in experiment 66. The direct
    float64 convolution builds a ~1681 x H x W im2col buffer on CPU and does not fit this machine at 512x512; --part check
    compares the two forms."""
    k = kernel[:, 0].double(); pad = k.shape[-1] // 2; H, Wd = x.shape[-2:]
    rows = torch.arange(-pad, H + pad).clamp(0, H - 1); cols = torch.arange(-pad, Wd + pad).clamp(0, Wd - 1)
    xp = x.double().index_select(-2, rows).index_select(-1, cols); Hp, Wp = xp.shape[-2:]
    full = torch.fft.irfft2(torch.fft.rfft2(xp, s=(Hp, Wp)) * torch.fft.rfft2(torch.flip(k, (-2, -1)), s=(Hp, Wp))[None], s=(Hp, Wp))
    return full[..., 2 * pad:2 * pad + H, 2 * pad:2 * pad + Wd]


class FWald:
    """54 prepare.Wald of a sensor (same kernels, phase 2, x4) with fft_blur."""

    def __init__(self, sensor):
        w = Wald(sensor, 'cpu'); self.ms_kernel, self.pan_kernel, self.max_value = w.ms_kernel, w.pan_kernel, w.max_value

    def ms(self, x): return fft_blur(x, self.ms_kernel)[..., P54.PHASE::4, P54.PHASE::4]
    def pan(self, x): return fft_blur(x, self.pan_kernel)[..., P54.PHASE::4, P54.PHASE::4]

    def degrade(self, pan_dn, ms_dn):
        p = torch.from_numpy(np.asarray(pan_dn, dtype=np.float64))[None]; m = torch.from_numpy(np.asarray(ms_dn, dtype=np.float64))[None]
        pan_b, ms_b = self.pan(p), self.ms(m)
        lms_b = P54.interp23tap((ms_b / self.max_value).float()).double() * self.max_value
        return pan_b[0].numpy(), lms_b[0].numpy(), ms_b[0].numpy()

    def observation_rmse(self, fused_dn, ms_dn, margin=5):
        e = self.ms(torch.from_numpy(np.asarray(fused_dn, dtype=np.float64))[None])[0].numpy() - np.asarray(ms_dn, dtype=np.float64)
        return float(np.sqrt(np.mean(e[..., margin:-margin, margin:-margin] ** 2)))


def lowpass(x, kernel):
    return fft_blur(torch.from_numpy(x)[None], kernel)[0].numpy()


def run_check(log):
    """FFT form vs the direct 54 operators on 128x128 crops of RR scene 0 (QB, WV2): blur, degrade and observation."""
    res = {}
    for target in ('QB', 'WV2'):
        a = scenes(target, 'RR')[0]; w, f = Wald(target, 'cpu'), FWald(target)
        x = torch.from_numpy(a['gt'][:, :128, :128])[None]; p = torch.from_numpy(a['pan'][:, :128, :128])[None]
        direct = w.degrade(a['pan'][:, :128, :128], a['gt'][:, :128, :128], 'cpu'); fast = f.degrade(a['pan'][:, :128, :128], a['gt'][:, :128, :128])
        res[target] = dict(ms_blur=float((_blur(x, w.ms_kernel) - fft_blur(x, w.ms_kernel)).abs().max()),
                           pan_blur=float((_blur(p, w.pan_kernel) - fft_blur(p, w.pan_kernel)).abs().max()),
                           degrade=[float(np.abs(d - q).max()) for d, q in zip(direct, fast)],
                           observation_rmse=abs(w.observation_rmse(a['gt'][:, :128, :128], direct[2], 'cpu') - f.observation_rmse(a['gt'][:, :128, :128], direct[2])))
    (OUT / 'fft_check.json').write_text(json.dumps(res, indent=1), encoding='utf-8'); log(json.dumps(res))
    return res


def split_mag_dir(h):
    m = np.linalg.norm(h, axis=0, keepdims=True)
    return m, np.where(m > 0, h / np.where(m > 0, m, 1), 0)


def shapley_split(g, ya, yb, la, lb, lg=None):
    """Change S(ya) - S(yb) (positive = yb better) split into LF/HF and HF into magnitude/ratio; also PSNR LF/HF.
    la, lb are the low bands L(ya), L(yb)."""
    ha, hb = ya - la, yb - lb
    S = {}
    for kl, L in (('a', la), ('b', lb)):
        for kh, H in (('a', ha), ('b', hb)): S[kl + kh] = sam_psnr(g, L + H)
    out = {}
    for i, name in ((0, 'SAM'), (1, 'PSNR')):
        s = {k: v[i] for k, v in S.items()}
        out[name] = dict(total=s['aa'] - s['bb'], LF=0.5 * ((s['aa'] - s['ba']) + (s['ab'] - s['bb'])),
                         HF=0.5 * ((s['aa'] - s['ab']) + (s['ba'] - s['bb'])))
    (ma, ua), (mb, ub) = split_mag_dir(ha), split_mag_dir(hb); mag = ratio = 0.
    for L in (la, lb):  # HF of each L split by the magnitude and the band ratio, averaged over the two L
        t = {k: sam_psnr(g, L + m * u)[0] for k, (m, u) in dict(aa=(ma, ua), ba=(mb, ua), ab=(ma, ub), bb=(mb, ub)).items()}
        mag += 0.5 * 0.5 * ((t['aa'] - t['ba']) + (t['ab'] - t['bb'])); ratio += 0.5 * 0.5 * ((t['aa'] - t['ab']) + (t['ba'] - t['bb']))
    out['SAM']['HF_magnitude'], out['SAM']['HF_band_ratio'] = mag, ratio
    if lg is not None:
        out['LF_fixed_SAM'] = dict(a=sam_psnr(g, lg + ha)[0], b=sam_psnr(g, lg + hb)[0])
    return out


def mean_tree(rows):
    if isinstance(rows[0], dict): return {k: mean_tree([r[k] for r in rows if k in r]) for k in rows[0]}
    if isinstance(rows[0], (int, float)): return float(np.mean(rows))
    return rows[0]


# ---------------------------------------------------------------- part: test

def run_test(target, log):
    sensor = CFG['targets'][target]; scale = SENSORS[target]['max_value']; W = FWald(target)
    kernel = torch.from_numpy(np.stack([gen_mtf(gn, 4) for gn in SENSORS[target]['ms_gnyq']])[:, None]).double()
    rr, fr = scenes(target, 'RR'), scenes(target, 'FR')
    nets, info = {}, {}
    for seed in SEEDS:
        c, info[f'C2MS_s{seed}'] = load_net('C2MS', sensor, seed)
        l, info[f'L1S_s{seed}'] = load_net('L1S', sensor, seed)
        p, _ = pruned(c)  # copied before the switchable _attend and the hooks are installed on c
        nets[seed] = dict(A=add_modes(c), PRUNED=p, FOLDED=Folded(c).eval(), L1S=l)
    full = ('RR', 'RR2', 'FR_WALD', 'FR')
    plan = [('A', full), ('FOLDED', full), ('L1S', full), ('PRUNED', ('RR', 'FR') if target == 'QB' else ('RR',)),
            ('Z_SSA', ('RR', 'FR') if target == 'QB' else ('RR',)), ('Z_STAGE', ('RR',))]
    ssa_max = {seed: [0.] * 3 for seed in SEEDS}
    hooks = []
    for seed in SEEDS:
        for k, (blocks, _, _) in enumerate(STAGES):
            for b in getattr(nets[seed]['A'].core, blocks):
                def hook(_m, _i, o, seed=seed, k=k):
                    if nets[seed]['A'].core.diag_mode == 'A' and _record[0]: ssa_max[seed][k] = max(ssa_max[seed][k], float(o.abs().max()))
                hooks.append(b.register_forward_hook(hook))
    _record = [False]
    rows = {seed: [] for seed in SEEDS}; ens_rows = []; started = time.time()
    for i, (a, f) in enumerate(zip(rr, fr)):
        x = {'RR': (a['pan'], a['lms'], a['ms']), 'RR2': W.degrade(a['pan'], a['ms']),
             'FR_WALD': W.degrade(f['pan'], f['ms']), 'FR': (f['pan'], f['lms'], f['ms'])}
        gt = {'RR': a['gt'], 'RR2': a['ms'], 'FR_WALD': f['ms']}
        preds = {}
        for seed in SEEDS:
            n = nets[seed]
            for variant, tests in plan:
                for t in tests:
                    if variant in ('A', 'Z_SSA', 'Z_STAGE'):
                        n['A'].core.diag_mode = variant; _record[0] = variant == 'A' and t == 'RR'
                        y = predict(n['A'], x[t], scale); n['A'].core.diag_mode = 'A'; _record[0] = False
                    else:
                        y = predict(n[variant], x[t], scale)
                    preds[seed, variant, t] = y
        lg = lowpass(a['gt'], kernel)
        low = {(s, v): lowpass(preds[s, v, 'RR'], kernel) for s in SEEDS for v in ('A', 'L1S')}
        for seed in SEEDS:
            r = dict(scene=i, metrics={}, d1={}, d2=None)
            for variant, tests in plan:
                for t in tests:
                    y = preds[seed, variant, t]
                    if t == 'FR': m = dict(OBS_RMSE_DN=W.observation_rmse(y, f['ms']))
                    elif variant in ('A', 'FOLDED', 'L1S'): m = rr_metrics(gt[t], y)
                    else:
                        s, p = sam_psnr(gt[t], y); m = dict(SAM=s, PSNR=p)
                    r['metrics'].setdefault(variant, {})[t] = m
                    if variant in ('PRUNED', 'Z_SSA', 'FOLDED', 'Z_STAGE'):
                        d = np.abs(y - preds[seed, 'A', t]); r['d1'].setdefault(variant, {})[t] = dict(max_abs_DN=float(d.max()), mean_abs_DN=float(d.mean()))
            fz = np.abs(preds[seed, 'FOLDED', 'RR'] - preds[seed, 'Z_SSA', 'RR']).max(); r['d1']['FOLDED_vs_Z_SSA_RR_max_abs_DN'] = float(fz)
            s, p = sam_psnr(a['gt'], preds[seed, 'A', 'RR']); ref = r['metrics']['A']['RR']
            r['sam_psnr_check'] = max(abs(s - ref['SAM']), abs(p - ref['PSNR']))
            r['d2'] = shapley_split(a['gt'], preds[seed, 'A', 'RR'], preds[seed, 'L1S', 'RR'], low[seed, 'A'], low[seed, 'L1S'], lg)
            rows[seed].append(r)
        e = dict(scene=i, metrics={}, d3_split={})
        members = {'E_C2MS': [(s, 'A') for s in SEEDS], 'E_L1S': [(s, 'L1S') for s in SEEDS]}
        members['E_ALL'] = members['E_C2MS'] + members['E_L1S']
        for name, mem in members.items():
            for t in full:
                y = np.mean([preds[s, v, t] for s, v in mem], axis=0)
                e['metrics'].setdefault(name, {})[t] = dict(OBS_RMSE_DN=W.observation_rmse(y, f['ms'])) if t == 'FR' else rr_metrics(gt[t], y)
                if t == 'RR':
                    ly = lowpass(y, kernel)
                    splits = [shapley_split(a['gt'], preds[s, v, 'RR'], y, low[s, v], ly) for s, v in mem]
                    e['d3_split'][name] = mean_tree(splits)
        ens_rows.append(e)
        log(f'{target} scene {i + 1}/20 {time.time() - started:.0f}s')
    for h in hooks: h.remove()
    res = dict(target=target, training_sensor=sensor, models=info, plan=[dict(variant=v, tests=list(t)) for v, t in plan],
               ssa_block_max_abs_normalized={str(s): ssa_max[s] for s in SEEDS},
               parameters=param_table(nets[SEEDS[0]]['A']),
               seeds={str(s): dict(mean=mean_tree([{k: v for k, v in r.items() if k != 'scene'} for r in rows[s]]), scenes=rows[s]) for s in SEEDS},
               ensembles=dict(mean=mean_tree([{k: v for k, v in r.items() if k != 'scene'} for r in ens_rows]), scenes=ens_rows),
               environment=dict(torch=torch.__version__, numpy=np.__version__, device='cpu', threads=torch.get_num_threads(),
                                seconds=round(time.time() - started, 1), finished=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    res['reproduction'] = reproduction(target, sensor, res)
    (OUT / f'raw_{target}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    bad = [k for k, v in res['reproduction'].items() if not v['ok']]
    if bad: raise ValueError(f'{target}: archived means not reproduced: {bad}')
    return res


def archived(target, sensor, seed, family):
    if family == 'C2MS' and target == 'QB':
        m = json.loads((PERF / '53_scale_mixed_training/results/evaluation_v2_3.json').read_text(encoding='utf-8'))['models'][f'C2MS_s{seed}']['mean']
    elif family == 'C2MS':
        m = json.loads((E54 / 'results/evaluation_v2_3.json').read_text(encoding='utf-8'))['models'][f'C2MS_{sensor}_s{seed}']['targets'][target]['mean']
    else:
        p = PERF / f'65_l1_spectral_loss/results/C2MS_L1S_{sensor}_s{seed}/evaluation/{target}_v2_3.json'
        m = json.loads(p.read_text(encoding='utf-8'))['means']['C2MS_L1S']
    return m


def reproduction(target, sensor, res):
    tol = {'RR': 1e-3, 'RR2': 5e-3, 'FR_WALD': 5e-3}; out = {}
    for seed in SEEDS:
        for family, variant in (('C2MS', 'A'), ('L1S', 'L1S')):
            ref = archived(target, sensor, seed, family); mine = res['seeds'][str(seed)]['mean']['metrics'][variant]
            diff = {f'{t}_{k}': mine[t][k] - ref[t][k] for t in tol for k in ('PSNR', 'SAM')}
            diff['FR_OBS_RMSE_DN'] = mine['FR']['OBS_RMSE_DN'] - ref['FR']['OBS_RMSE_DN']
            ok = all(abs(diff[f'{t}_{k}']) <= tol[t] for t in tol for k in ('PSNR', 'SAM')) and abs(diff['FR_OBS_RMSE_DN']) <= 0.05
            out[f'{family}_s{seed}'] = dict(ok=bool(ok), diff=diff)
    return out


# ---------------------------------------------------------------- part: validation (QB)

def run_val(log, batch=64):
    path = ROOT / 'data/validation/valid_qb.h5'
    if sha(path) != CFG['data_sha256']['QB']['val']: raise ValueError('validation sha mismatch')
    nets = {}
    for seed in SEEDS:
        c, _ = load_net('C2MS', 'QB', seed); nets[f'C2MS_s{seed}'] = c; nets[f'FOLDED_s{seed}'] = Folded(c)
        nets[f'L1S_s{seed}'], _ = load_net('L1S', 'QB', seed)
    groups = {'E_C2MS': [f'C2MS_s{s}' for s in SEEDS], 'E_L1S': [f'L1S_s{s}' for s in SEEDS]}
    groups['E_ALL'] = groups['E_C2MS'] + groups['E_L1S']
    names = list(nets) + list(groups)
    acc = {k: np.zeros(3) for k in names}; pixels = {k: 0 for k in names}; elements = 0; n = 0
    folded_max = {s: 0. for s in SEEDS}; started = time.time()
    with h5py.File(path, 'r') as f:
        total = len(f['pan'])
        for i in range(0, total, batch):
            x = [torch.from_numpy((np.asarray(f[k][i:i + batch]) / 2047.).astype('float32')) for k in ('pan', 'lms', 'ms')]
            gt = torch.from_numpy((np.asarray(f['gt'][i:i + batch]) / 2047.).astype('float32'))
            out = {k: net(*x).float() for k, net in nets.items()}
            for k, mem in groups.items(): out[k] = torch.stack([out[m] for m in mem]).mean(0)
            for s in SEEDS: folded_max[s] = max(folded_max[s], float((out[f'FOLDED_s{s}'] - out[f'C2MS_s{s}']).abs().max()) * 2047)
            for k in names:  # same accumulation as the 53/54 trainer validation()
                p = out[k]; err = (p - gt).square()
                acc[k][0] += err.double().sum().item()
                acc[k][1] += (-10 * torch.log10(err.mean((2, 3)).clamp_min(1e-15))).mean(1).double().sum().item()
                norm = p.norm(dim=1) * gt.norm(dim=1); mask = norm > 1e-12; cos = (p * gt).sum(1) / norm.clamp_min(1e-12)
                acc[k][2] += torch.acos(cos.clamp(-1, 1))[mask].double().sum().item(); pixels[k] += int(mask.sum())
            elements += gt.numel(); n += len(gt)
            log(f'val {min(i + batch, total)}/{total} {time.time() - started:.0f}s')
    val = {k: dict(val_mse=acc[k][0] / elements, val_psnr=acc[k][1] / n, val_sam=float(np.degrees(acc[k][2] / max(pixels[k], 1)))) for k in names}
    archived_val = {}
    for s in SEEDS:
        for fam, key in (('C2MS', f'C2MS_s{s}'), ('L1S', f'L1S_s{s}')):
            ck = torch.load(ROOT / CFG['checkpoints'][fam]['QB'].replace('<seed>', str(s)), map_location='cpu', weights_only=False)['metrics']
            archived_val[key] = {k: val[key][k] - ck[k] for k in ('val_psnr', 'val_sam')}
    res = dict(metrics=val, folded_vs_A_max_abs_DN={str(s): v for s, v in folded_max.items()},
               reproduction_vs_checkpoint_metrics=archived_val, patches=n,
               environment=dict(torch=torch.__version__, device='cpu', threads=torch.get_num_threads(), seconds=round(time.time() - started, 1),
                                finished=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    (OUT / 'validation_QB.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    return res


# ---------------------------------------------------------------- part: latency

def run_latency(log, repeats=10, warmup=3):
    net, _ = load_net('C2MS', 'QB', 42); models = {'A': net, 'FOLDED': Folded(net)}
    devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
    res = {}
    for dev in devices:
        sync = torch.mps.synchronize if dev == 'mps' else (lambda: None)
        for name, m in models.items():
            m = copy.deepcopy(m).to(dev)
            for b, s in ((1, 256), (1, 512), (32, 64)):
                g = torch.Generator().manual_seed(0)
                x = [torch.rand(b, c, h, h, generator=g).to(dev) for c, h in ((1, s), (4, s), (4, s // 4))]
                for _ in range(warmup): m(*x)
                sync(); times = []
                for _ in range(repeats):
                    t = time.perf_counter(); m(*x); sync(); times.append(time.perf_counter() - t)
                res[f'{dev}/{name}/b{b}x{s}'] = dict(median_s=float(np.median(times)), min_s=float(np.min(times)), runs=repeats)
                log(f'latency {dev} {name} b{b}x{s} {np.median(times) * 1000:.1f} ms')
    res['environment'] = dict(torch=torch.__version__, cpu_threads=torch.get_num_threads(), devices=devices, note='random inputs; seed 42 weights')
    (OUT / 'latency.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    return res


# ---------------------------------------------------------------- part: images

def run_images(log):
    base, _ = load_net('C2MS', 'QB', 42)
    ens = Ensemble([load_net('C2MS', 'QB', s)[0] for s in SEEDS]).eval()
    stats = result_images.render(base, ens, ROOT / 'data', OUT / 'images/C2MS_vs_E_C2MS', names=('C2MS', 'E_C2MS'), seed=42, device='cpu',
                                 labels=('현재 기준 C2MS (seed 42)', 'C2MS 3시드 앙상블'))
    log(f'images {sorted(stats)}')


# ---------------------------------------------------------------- part: summary

def seed_mean(res, getter):
    return float(np.mean([getter(res['seeds'][str(s)]['mean']) for s in SEEDS]))


def summarize(log):
    raw = {t: json.loads((OUT / f'raw_{t}.json').read_text(encoding='utf-8')) for t in CFG['targets'] if (OUT / f'raw_{t}.json').exists()}
    val = json.loads((OUT / 'validation_QB.json').read_text(encoding='utf-8')) if (OUT / 'validation_QB.json').exists() else None
    lat = json.loads((OUT / 'latency.json').read_text(encoding='utf-8')) if (OUT / 'latency.json').exists() else None
    s = dict(targets={})
    gate1 = True
    for t, r in raw.items():
        d = dict()
        tests = ('RR', 'RR2', 'FR_WALD', 'FR')
        for v in ('PRUNED', 'Z_SSA', 'FOLDED', 'Z_STAGE'):
            vt = [x['tests'] for x in r['plan'] if x['variant'] == v][0]
            d[v] = {}
            for test in vt:
                mx = max(sc['d1'][v][test]['max_abs_DN'] for sd in r['seeds'].values() for sc in sd['scenes'])
                row = dict(max_abs_DN=mx)
                if test == 'FR':
                    row['obs_ratio'] = seed_mean(r, lambda m: m['metrics'][v]['FR']['OBS_RMSE_DN'] / m['metrics']['A']['FR']['OBS_RMSE_DN'])
                else:
                    row['dPSNR'] = seed_mean(r, lambda m: m['metrics'][v][test]['PSNR'] - m['metrics']['A'][test]['PSNR'])
                    row['dSAM'] = seed_mean(r, lambda m: m['metrics'][v][test]['SAM'] - m['metrics']['A'][test]['SAM'])
                d[v][test] = row
        g = all(d['FOLDED'][x]['max_abs_DN'] <= 0.5 for x in tests) and all(abs(d['FOLDED'][x]['dPSNR']) <= 1e-3 and abs(d['FOLDED'][x]['dSAM']) <= 1e-3 for x in tests[:3]) \
            and abs(d['FOLDED']['FR']['obs_ratio'] - 1) <= 1e-3 and all(v['max_abs_DN'] == 0 for v in d['PRUNED'].values())
        gate1 &= g
        a = lambda key: (lambda m: m['metrics']['A'][key[0]][key[1]])
        l = lambda key: (lambda m: m['metrics']['L1S'][key[0]][key[1]])
        d2 = {k: seed_mean(r, lambda m, k=k: m['d2']['SAM'][k]) for k in ('total', 'LF', 'HF', 'HF_magnitude', 'HF_band_ratio')}
        d2['PSNR'] = {k: -seed_mean(r, lambda m, k=k: m['d2']['PSNR'][k]) for k in ('total', 'LF', 'HF')}  # positive = 65 better
        d2['LF_share'] = d2['LF'] / d2['total'] if d2['total'] else None
        d2['C2MS_SAM'] = seed_mean(r, a(('RR', 'SAM'))); d2['L1S_SAM'] = seed_mean(r, l(('RR', 'SAM')))
        d2['L1S_LF_fixed_SAM'] = seed_mean(r, lambda m: m['d2']['LF_fixed_SAM']['b']); d2['C2MS_LF_fixed_SAM'] = seed_mean(r, lambda m: m['d2']['LF_fixed_SAM']['a'])
        d2['L1S_remaining_LF_headroom'] = d2['L1S_SAM'] - d2['L1S_LF_fixed_SAM']; d2['C2MS_LF_headroom'] = d2['C2MS_SAM'] - d2['C2MS_LF_fixed_SAM']
        singles = {fam: {test: {k: seed_mean(r, lambda m, fam=fam, test=test, k=k: m['metrics'][fam][test][k]) for k in ('PSNR', 'SAM', 'ERGAS')}
                         for test in tests[:3]} for fam in ('A', 'L1S')}
        for fam in ('A', 'L1S'): singles[fam]['FR'] = dict(OBS_RMSE_DN=seed_mean(r, lambda m, fam=fam: m['metrics'][fam]['FR']['OBS_RMSE_DN']))
        em = r['ensembles']['mean']['metrics']; d3 = {}
        for name, ref in (('E_C2MS', 'A'), ('E_L1S', 'L1S'), ('E_ALL', None)):
            refm = singles[ref] if ref else {test: {k: 0.5 * (singles['A'][test][k] + singles['L1S'][test][k]) for k in singles['A'][test]} for test in tests}
            d3[name] = {test: ({k: em[name][test][k] - refm[test][k] for k in ('PSNR', 'SAM', 'ERGAS')} if test != 'FR'
                               else dict(obs_ratio=em[name]['FR']['OBS_RMSE_DN'] / refm['FR']['OBS_RMSE_DN'])) for test in tests}
            d3[name]['vs_C2MS_mean'] = {k: em[name]['RR'][k] - singles['A']['RR'][k] for k in ('PSNR', 'SAM', 'ERGAS')}
            d3[name]['absolute_RR'] = {k: em[name]['RR'][k] for k in ('PSNR', 'SAM', 'ERGAS')}
            sp = r['ensembles']['mean']['d3_split'][name]['SAM']
            d3[name]['RR_SAM_split_vs_members'] = {k: sp[k] for k in ('total', 'LF', 'HF', 'HF_magnitude', 'HF_band_ratio')}
        scs = [sc for sd in r['seeds'].values() for sc in sd['scenes']]
        d['FOLDED_vs_Z_SSA_RR_max_abs_DN'] = max(sc['d1']['FOLDED_vs_Z_SSA_RR_max_abs_DN'] for sc in scs)
        d['sam_psnr_vs_rr_metrics_max_abs'] = max(sc['sam_psnr_check'] for sc in scs)
        s['targets'][t] = dict(D1=d, D1_pass=bool(g), D2=d2, D3=d3, singles=singles, parameters=r['parameters'],
                               ssa_block_max_abs_DN={k: [x * SENSORS[t]['max_value'] for x in v] for k, v in r['ssa_block_max_abs_normalized'].items()},
                               reproduction_ok=all(v['ok'] for v in r['reproduction'].values()))
    if val:
        vm = val['metrics']; dv = {}
        for name, mem in (('E_C2MS', 'C2MS'), ('E_L1S', 'L1S')):
            ref = {k: float(np.mean([vm[f'{mem}_s{x}'][k] for x in SEEDS])) for k in ('val_psnr', 'val_sam', 'val_mse')}
            dv[name] = dict(dPSNR=vm[name]['val_psnr'] - ref['val_psnr'], dSAM=vm[name]['val_sam'] - ref['val_sam'], members_mean=ref, ensemble=vm[name])
        ref = {k: float(np.mean([vm[f'{m}_s{x}'][k] for m in ('C2MS', 'L1S') for x in SEEDS])) for k in ('val_psnr', 'val_sam')}
        dv['E_ALL'] = dict(dPSNR=vm['E_ALL']['val_psnr'] - ref['val_psnr'], dSAM=vm['E_ALL']['val_sam'] - ref['val_sam'])
        s['validation_QB'] = dict(ensembles=dv, folded_vs_A_max_abs_DN=val['folded_vs_A_max_abs_DN'],
                                  folded_dPSNR={x: vm[f'FOLDED_s{x}']['val_psnr'] - vm[f'C2MS_s{x}']['val_psnr'] for x in map(str, SEEDS)},
                                  reproduction_vs_checkpoint_metrics=val['reproduction_vs_checkpoint_metrics'])
        gate1 &= max(val['folded_vs_A_max_abs_DN'].values()) <= 0.5
    gates = {}
    if raw: gates['D1'] = dict(passed=bool(gate1), decision='SSA blocks and unused layers removable without retraining' if gate1 else 'SSA contribution not negligible; no removal')
    if 'QB' in s['targets']:
        q = s['targets']['QB']['D2']; share = q['LF_share']
        verdict = 'LF_exhausted' if share is not None and share >= 0.70 and q['L1S_remaining_LF_headroom'] <= 0.03 else ('HF_driven' if share is not None and 1 - share >= 0.50 else 'mixed')
        gates['D2'] = dict(LF_share=share, HF_share=None if share is None else 1 - share, remaining_LF_headroom=q['L1S_remaining_LF_headroom'], verdict=verdict)
    if val:
        e = s['validation_QB']['ensembles']
        hit = any(e[k]['dPSNR'] >= 0.10 or e[k]['dSAM'] <= -0.05 for k in ('E_C2MS', 'E_L1S'))
        gates['D3'] = dict(passed=bool(hit), verdict='variance_headroom' if hit else 'small', E_C2MS=e['E_C2MS'], E_L1S=e['E_L1S'])
    s['gates'] = gates; s['latency'] = lat
    (OUT / 'summary.json').write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
    plot(s)
    log(json.dumps(gates, ensure_ascii=False, indent=1))
    return s


def plot(s):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    targets = list(s['targets']); fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    tests = ('RR', 'RR2', 'FR_WALD', 'FR'); w = 0.8 / len(targets)
    for i, t in enumerate(targets):
        vals = [max(s['targets'][t]['D1']['FOLDED'][x]['max_abs_DN'], 1e-7) for x in tests]
        ax[0, 0].bar(np.arange(4) + (i - (len(targets) - 1) / 2) * w, vals, w, label=t)
    ax[0, 0].axhline(0.5, color='k', ls='--', lw=1, label='gate 0.5 DN'); ax[0, 0].set_yscale('log'); ax[0, 0].set_xticks(range(4), tests)
    ax[0, 0].set_ylabel('max |C2MS - FOLDED| over all pixels (DN), 3 seeds'); ax[0, 0].set_title('D1 SSAI folded to constants vs C2MS'); ax[0, 0].legend(fontsize=8)
    if s.get('latency'):
        lat = s['latency']; keys = [k for k in lat if k.count('/') == 2]; groups = sorted({k.split('/')[0] + ' ' + k.split('/')[2] for k in keys})
        for j, name in enumerate(('A', 'FOLDED')):
            v = [lat[f"{g.split()[0]}/{name}/{g.split()[1]}"]['median_s'] * 1000 for g in groups]
            ax[0, 1].bar(np.arange(len(groups)) + (j - 0.5) * 0.4, v, 0.4, label={'A': 'C2MS', 'FOLDED': 'FOLDED'}[name])
        ax[0, 1].set_xticks(range(len(groups)), groups, rotation=20, fontsize=8); ax[0, 1].set_yscale('log'); ax[0, 1].set_ylabel('median forward time (ms)')
        ax[0, 1].set_title('D1 latency (seed 42 weights, random input)'); ax[0, 1].legend(fontsize=8)
    for col, (key, parts, label) in ((2, ('SAM', ('total', 'LF', 'HF_magnitude', 'HF_band_ratio'), 'SAM decrease (deg)')),
                                     (None, ('PSNR', ('total', 'LF', 'HF'), 'PSNR gain (dB)'))):
        a = ax[0, 2] if col == 2 else ax[1, 2]; wd = 0.8 / len(parts)
        for j, k in enumerate(parts):
            vals = [(s['targets'][t]['D2'][k] if key == 'SAM' else s['targets'][t]['D2']['PSNR'][k]) for t in targets]
            a.bar(np.arange(len(targets)) + (j - (len(parts) - 1) / 2) * wd, vals, wd, label=k)
        a.axhline(0, color='k', lw=0.8); a.set_xticks(range(len(targets)), targets); a.legend(fontsize=8)
        a.set_ylabel(f'65 vs C2MS: {label}, Shapley parts'); a.set_title(f'D2 {key} change of 65 split by frequency (RR test, 3 seeds)')
    names = ('E_C2MS', 'E_L1S', 'E_ALL'); rows = []
    if s.get('validation_QB'): rows.append(('QB val', {n: s['validation_QB']['ensembles'][n] for n in names}))
    for t in targets: rows.append((f'{t} RR', {n: dict(dPSNR=s['targets'][t]['D3'][n]['RR']['PSNR'], dSAM=s['targets'][t]['D3'][n]['RR']['SAM']) for n in names}))
    for col, (k, label, gate) in enumerate((('dPSNR', 'PSNR (dB)', 0.10), ('dSAM', 'SAM (deg)', -0.05))):
        a = ax[1, col]
        for j, n in enumerate(names):
            a.bar(np.arange(len(rows)) + (j - 1) * 0.27, [r[1][n][k] for r in rows], 0.27, label=n)
        a.axhline(gate, color='gray', ls='--', lw=0.8, label=f'validation gate {gate:+.2f}'); a.axhline(0, color='k', lw=0.8)
        a.set_xticks(range(len(rows)), [r[0] for r in rows]); a.set_ylabel(f'ensemble - mean of its members: {label}')
        a.set_title(f'D3 seed ensembles, {label}'); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'diagnostics.png', dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', nargs='+', default=['test', 'val', 'latency', 'images', 'summary'],
                   choices=['check', 'test', 'val', 'latency', 'images', 'summary'])
    p.add_argument('--targets', nargs='+', default=list(CFG['targets']))
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    log = lambda msg: print(time.strftime('%H:%M:%S'), msg, flush=True)
    if 'check' in a.part: run_check(log)
    if 'test' in a.part:
        for t in a.targets: run_test(t, log)
    if 'val' in a.part: run_val(log)
    if 'latency' in a.part: run_latency(log)
    if 'images' in a.part: run_images(log)
    if 'summary' in a.part: summarize(log)


if __name__ == '__main__':
    main()
