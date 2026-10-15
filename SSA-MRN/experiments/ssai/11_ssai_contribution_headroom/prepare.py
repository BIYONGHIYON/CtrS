"""SSAI 11 phase 0 (no training): D6 oracle headroom of coefficient x PAN-basis injection,
D7 validation FR-proxy calibration against models with known FR test results."""
import os, sys, json, time, importlib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = r'C:\CtrS-ssai11\repo\SSA-MRN'
sys.path.insert(0, REPO)
import numpy as np, torch
from torch.nn import functional as F
from common import ssai_suite as suite
from common.model import RestoredPansharpeningNet
from common.metrics import fr_metrics

OUT = r'C:\CtrS-ssai11\runtime\phase0.json'
CACHE = r'C:\CtrS-ssai4\runtime\qb_s42_100\cache\val'
A0 = r'C:\CtrS-ssai4\runtime\qb_s42_100\A0\best.pt'


def down_up(x, factor):
    h, w = x.shape[-2:]
    d = F.interpolate(x, size=(h // factor, w // factor), mode='bicubic', align_corners=False, antialias=True)
    return F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)


def bases(pan, kind):
    lp4 = down_up(pan, 4)
    one = torch.ones_like(pan)
    if kind == 'S1':
        return torch.cat([one, pan - lp4], 1)
    lp2 = down_up(pan, 2)
    gx = F.pad(pan[..., :, 2:] - pan[..., :, :-2], (1, 1, 0, 0), mode='replicate') / 2
    gy = F.pad(pan[..., 2:, :] - pan[..., :-2, :], (0, 0, 1, 1), mode='replicate') / 2
    return torch.cat([one, pan - lp4, pan - lp2, lp2 - lp4, gx, gy], 1)


def oracle_explained(err, B, c):
    """Fraction of err energy removed by per-band, per-cell least-squares fit on bases B."""
    n, nb, h, w = err.shape; k = B.shape[1]
    Bc = B.reshape(n, k, h // c, c, w // c, c).permute(0, 2, 4, 3, 5, 1).reshape(n, -1, c * c, k).double()
    ata = Bc.transpose(-1, -2) @ Bc
    ata = ata + 1e-10 * torch.eye(k, dtype=ata.dtype, device=ata.device) * ata.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None].clamp_min(1e-12)
    resid = 0.
    for b in range(nb):
        e = err[:, b:b+1].reshape(n, 1, h // c, c, w // c, c).permute(0, 2, 4, 3, 5, 1).reshape(n, -1, c * c, 1).double()
        coef = torch.linalg.solve(ata, Bc.transpose(-1, -2) @ e)
        resid += float(((e - Bc @ coef) ** 2).sum())
    return 1 - resid / float((err.double() ** 2).sum())


def predict(net, pan, lms, ms):
    out = []
    with torch.inference_mode():
        for i in range(0, len(pan), 64):
            out.append(net(pan[i:i+64].cuda(), lms[i:i+64].cuda(), ms[i:i+64].cuda()).float().cpu())
    return torch.cat(out)


def proxy(pred, pan, lms, ms, idx):
    rows = [fr_metrics(pred[i].double().numpy() * 2047, lms[i].double().numpy() * 2047, ms[i].double().numpy() * 2047, pan[i, 0].double().numpy() * 2047) for i in idx]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    suite.configure(42, 'cuda')
    va = suite.Cached(CACHE)
    pan, lms, ms, gt = (torch.from_numpy(np.array(va.arrays[k])) for k in ('pan', 'lms', 'ms', 'gt'))
    a0 = RestoredPansharpeningNet(4, 4); a0.load_state_dict(torch.load(A0, map_location='cpu', weights_only=False)['model']); a0.cuda().eval()
    pred = predict(a0, pan, lms, ms)
    out = {'D6': {}, 'D7': {}}
    for name, base in (('A0', pred), ('LMS', lms)):
        err = (gt - base).cuda()
        rows = {}
        for kind in ('S1', 'S2'):
            B = torch.cat([bases(pan[i:i+256].cuda(), kind) for i in range(0, len(pan), 256)])
            for c in (4, 8, 16, 64):
                rows[f'{kind}_cell{c}'] = oracle_explained(err, B, c)
        out['D6'][name] = {'mse': float((err.double() ** 2).mean()), 'explained_fraction': rows}
        print(name, json.dumps(rows), flush=True)
    idx = list(range(0, len(pan), len(pan) // 256))[:256]
    models = []

    def add(name, net, ck, test_ds, test_qnr):
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        net.load_state_dict(sd['model'], strict=True); net.cuda().eval()
        p = predict(net, pan, lms, ms)
        models.append({'name': name, 'val_mse': float(((p - gt).double() ** 2).mean()), 'proxy': proxy(p, pan, lms, ms, idx), 'test_Ds': test_ds, 'test_QNR': test_qnr})
        print(name, 'done', flush=True)

    R = REPO
    s8 = json.load(open(os.path.join(R, r'experiments\ssai\8_spatial_mean_normalization\results\evaluation\summary.json')))['means']
    s9 = json.load(open(os.path.join(R, r'experiments\ssai\9_pan_innovation_correction\results\evaluation\summary.json')))['means']
    add('A0', RestoredPansharpeningNet(4, 4), A0, s8['A0']['FR']['D_s'], s8['A0']['FR']['QNR'])
    M8 = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.model')
    for arm, g in (('N1', False), ('N2', True)):
        add('ssai8_' + arm, M8.MeanNormalizedNet(4, 4, learn_gamma=g), rf'C:\CtrS-ssai8\runtime\qb_s42_100_verified\{arm}\best.pt', s8[arm]['FR']['D_s'], s8[arm]['FR']['QNR'])
    M9 = importlib.import_module('experiments.ssai.9_pan_innovation_correction.model')
    for arm in ('A1', 'A2'):
        add('ssai9_' + arm, M9.PanInnovationNet(4, 4, innovation=arm == 'A2'), rf'C:\CtrS-ssai10\runtime\qb_s42_100\{arm}\best.pt', s9[arm]['FR']['D_s'], s9[arm]['FR']['QNR'])
    M29 = importlib.import_module('experiments.performance.29_capacity_refinement.model')
    calib = json.load(open(r'C:\CtrS-p150\runtime\QB\plan.json'))['calibration']
    for target in (50, 150):
        m29 = json.load(open(os.path.join(R, rf'experiments\performance\29_capacity_refinement\results\QB\evaluation\{target}\metrics.json')))['runs']
        for v in ('R0', 'W2', 'W3', 'F01', 'F11'):
            fr = m29[v]['metrics']['FR']
            mean = fr['mean'] if 'mean' in fr else {k: float(np.mean([r['metrics'][k] for r in fr['rows']])) for k in fr['rows'][0]['metrics']}
            net = M29.StructureNet(v, 4); net.configure(calib)
            add(f'perf29_{v}@{target}', net, rf'C:\CtrS-p150\runtime\QB\{v}\best_at_{target}.pt', mean['D_s'], mean['QNR'])
    from scipy.stats import spearmanr, pearsonr
    ds_p = [m['proxy']['D_s'] for m in models]; ds_t = [m['test_Ds'] for m in models]
    q_p = [m['proxy']['QNR'] for m in models]; q_t = [m['test_QNR'] for m in models]
    out['D7'] = {'models': models, 'spearman_Ds': float(spearmanr(ds_p, ds_t)[0]), 'pearson_Ds': float(pearsonr(ds_p, ds_t)[0]),
                 'spearman_QNR': float(spearmanr(q_p, q_t)[0]), 'pearson_QNR': float(pearsonr(q_p, q_t)[0]), 'n': len(models), 'patches': 256}
    out['gates'] = {'D6_pass': out['D6']['A0']['explained_fraction']['S2_cell8'] >= 0.15, 'D6_stop': out['D6']['A0']['explained_fraction']['S2_cell8'] <= 0.05,
                    'D7_proxy_usable': out['D7']['spearman_Ds'] >= 0.7}
    out['time'] = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=2)
    print(json.dumps({k: v for k, v in out['D7'].items() if k != 'models'}), json.dumps(out['gates']))
    for m in models: print('%-18s val_mse=%.4e proxyDs=%.4f testDs=%.4f proxyQNR=%.4f testQNR=%.4f' % (m['name'], m['val_mse'], m['proxy']['D_s'], m['test_Ds'], m['proxy']['QNR'], m['test_QNR']))


if __name__ == '__main__':
    main()
