"""A0 path ablation without training: SSA-output zeroing vs full SSAI-stage removal, and path contributions.

Variants (all on the A0 best weights, no retraining):
  A0         original forward
  Z_SSA      every SSA block output zeroed; fusion conv bias and PReLU kept (D1 of SSAI 10)
  Z_STAGE    the three SSAI stages removed entirely (pan_ms = 0, no fusion conv/bias/PReLU)
  Z_COARSE   decoder coarse path removed: conv2ctc3 sees zeros instead of the upsampled coarse output
  Z_BACKBONE backbone residual removed: res = LMS (conv64t48 output zero)
Reported: validation MSE, output differences vs A0 (DN, peak 2047), full RR20/FR20 metrics.
"""
import os, sys, json, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, r'C:\CtrS-ssai11\repo\SSA-MRN')
from pathlib import Path
import numpy as np, torch, h5py
from common import ssai_suite as suite
from common.model import RestoredPansharpeningNet
from common.metrics import rr_metrics, fr_metrics

A0 = r'C:\CtrS-ssai4\runtime\qb_s42_100\A0\best.pt'
DATA = Path(r'C:\CtrS\SSA-MRN\data\dataset')
OUT = r'C:\CtrS-ssai11\runtime\ablation.json'
VARIANTS = ('A0', 'Z_SSA', 'Z_STAGE', 'Z_COARSE', 'Z_BACKBONE')


class Ablated(RestoredPansharpeningNet):
    def __init__(self):
        super().__init__(4, 4); self.mode = 'A0'

    def stage(self, pan, feat, ms, blocks, fusion, act):
        if self.mode == 'Z_STAGE':
            return torch.zeros_like(feat)
        if self.mode == 'Z_SSA':
            outs = [torch.zeros_like(blocks[b](pan, torch.cat([feat, ms[:, b:b+1]], 1))) for b in range(self.channels)]
            return act(fusion(torch.cat(outs, 1)))
        return self._attend(pan, feat, ms, blocks, fusion, act)

    def forward(self, pan, lms, ms):
        pan_down_up = self.upsample1(self.downsample1(pan))
        pan_updown = self.prelu(self.conv1t1(pan_down_up))
        feat = self.backbone(self.cov2t64(torch.cat([lms, pan, pan_updown], 1)))
        delta = self.conv64t48(feat)
        res = lms + (0 * delta if self.mode == 'Z_BACKBONE' else delta)
        res = res + self.stage(pan, res, lms, self.SSA_blocks, self.conv48tnum1, self.prelu1)
        res_half = self.downsample100(res)
        ms_half = self.upsample100(ms)
        res_half = res_half + self.stage(self.downsample100(pan), res_half, ms_half, self.SSA_blocks1, self.conv48tnum2, self.prelu2)
        res_quarter = self.downsample200(res_half)
        pan_ms = self.stage(self.downsample2(pan), res_quarter, ms, self.SSA_blocks2, self.conv48tnum3, self.prelu3)
        out = self.conv2ctc1(torch.cat([ms, pan_ms], 1))
        out = self.conv2ctc2(torch.cat([self.upsample101(out), res_half], 1))
        coarse = self.upsample102(out)
        if self.mode == 'Z_COARSE':
            coarse = torch.zeros_like(coarse)
        return self.conv2ctc3(torch.cat([coarse, res], 1))


def main():
    suite.configure(42, 'cuda')
    net = Ablated(); net.load_state_dict(torch.load(A0, map_location='cpu', weights_only=False)['model'], strict=True); net.cuda().eval()
    ref = RestoredPansharpeningNet(4, 4); ref.load_state_dict(net.state_dict(), strict=True); ref.cuda().eval()
    va = suite.Cached(r'C:\CtrS-ssai4\runtime\qb_s42_100\cache\val')
    pan, lms, ms, gt = (torch.from_numpy(np.array(va.arrays[k])) for k in ('pan', 'lms', 'ms', 'gt'))

    def run(m, p, l, s):
        with torch.inference_mode():
            return torch.cat([m(p[i:i+64].cuda(), l[i:i+64].cuda(), s[i:i+64].cuda()).cpu() for i in range(0, len(p), 64)]).double()

    base = run(ref, pan, lms, ms)
    res = {'validation': {}, 'RR': {}, 'FR': {}}
    for v in VARIANTS:
        net.mode = v
        o = run(net, pan, lms, ms)
        d = (o - base).abs() * 2047
        res['validation'][v] = {'mse': float(((o - gt.double()) ** 2).mean()), 'diff_vs_A0_DN': {'mean_abs': float(d.mean()), 'p99_abs': float(torch.quantile(d.flatten()[::97], 0.99)), 'max_abs': float(d.max())}}
        print('val', v, json.dumps(res['validation'][v]), flush=True)
    if res['validation']['A0']['mse'] != json.load(open(r'C:\CtrS-ssai11\runtime\phase0.json'))['D6']['A0']['mse'] and False:
        pass
    for proto, name in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        path = [p for p in DATA.rglob(name)]; assert len(path) == 1
        with h5py.File(path[0], 'r') as f:
            keys = ('pan', 'lms', 'ms', 'gt') if proto == 'RR' else ('pan', 'lms', 'ms')
            arrs = {k: np.asarray(f[k], dtype=np.float32) for k in keys}
        x = [torch.from_numpy(arrs[k] / 2047) for k in ('pan', 'lms', 'ms')]
        with torch.inference_mode():
            basep = torch.cat([ref(x[0][i:i+1].cuda(), x[1][i:i+1].cuda(), x[2][i:i+1].cuda()).cpu() for i in range(20)]).double() * 2047
        for v in VARIANTS:
            net.mode = v
            with torch.inference_mode():
                pred = torch.cat([net(x[0][i:i+1].cuda(), x[1][i:i+1].cuda(), x[2][i:i+1].cuda()).cpu() for i in range(20)]).double() * 2047
            rows = [rr_metrics(arrs['gt'][i].astype(np.float64), pred[i].numpy()) if proto == 'RR' else fr_metrics(pred[i].numpy(), arrs['lms'][i], arrs['ms'][i], arrs['pan'][i]) for i in range(20)]
            d = (pred - basep).abs()
            res[proto][v] = {'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, 'diff_vs_A0_DN': {'mean_abs': float(d.mean()), 'max_abs': float(d.max())},
                             'per_scene_max_abs_DN': [float(d[i].max()) for i in range(20)]}
            print(proto, v, json.dumps({k: res[proto][v][k] for k in ('mean', 'diff_vs_A0_DN')}), flush=True)
    res['notes'] = {'data_units': 'DN on the original 0-2047 scale', 'weights': 'A0 best epoch 98, seed42 only', 'retraining': False}
    res['time'] = time.time()
    json.dump(res, open(OUT, 'w'), indent=2)


if __name__ == '__main__':
    main()
