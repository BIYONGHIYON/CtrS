"""Structural, size, crop and checkpoint checks using synthetic data only."""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ssamrn.models.ssa_fusion import FusionPansharpeningNet, LocalCrossSSA
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet


def embedded_worker(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'WORKER_SOURCE':
            return ast.literal_eval(node.value)
    raise ValueError('Missing embedded worker')


def verify(output, device='cpu'):
    torch.set_num_threads(2)
    rows = {}
    torch.manual_seed(46)
    reference = RestoredPansharpeningNet(4, 6)
    shared = reference.state_dict()
    for variant in ('A0','A1','A2','A3'):
        torch.manual_seed(46)
        model = FusionPansharpeningNet(4, 6, variant).to(device)
        assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in shared.items())
        row = dict(parameters_total=sum(p.numel() for p in model.parameters()), shapes=[])
        for h,w in ((32,48),(64,64),(256,256),(512,512)):
            pan = torch.rand(1,1,h,w,device=device)
            lms = torch.rand(1,4,h,w,device=device)
            ms = torch.rand(1,4,h//4,w//4,device=device)
            model.eval()
            if device.startswith('cuda'):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            with torch.no_grad():
                pred = model(pan,lms,ms)
            if device.startswith('cuda'): torch.cuda.synchronize()
            assert pred.shape == lms.shape and torch.isfinite(pred).all()
            elapsed=time.perf_counter()-tick
            if variant == 'A0' and h == 32:
                with torch.no_grad(): assert torch.equal(pred.cpu(), reference.eval()(pan.cpu(),lms.cpu(),ms.cpu()))
            row['shapes'].append(dict(shape=[h,w], single_forward_seconds=elapsed, peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None))
            if h == 64:
                # End-to-end crop difference is measured, not assumed zero: original
                # align_corners interpolation also changes coordinates with size.
                with torch.no_grad():
                    crop = model(pan[...,16:48,16:48],lms[...,16:48,16:48],ms[...,4:12,4:12])
                row['full_vs_crop_interior_mae'] = (pred[...,24:40,24:40]-crop[...,8:24,8:24]).abs().mean().item()
        # Overlapping 48x48 tiles with 16px halo, compared in their 16x16 cores.
        pan=torch.rand(1,1,64,64,device=device)
        lms=torch.rand(1,4,64,64,device=device)
        ms=torch.rand(1,4,16,16,device=device)
        with torch.no_grad():
            full=model(pan,lms,ms)
            differences=[]
            for y,x in ((0,0),(0,16),(16,0),(16,16)):
                tile=model(pan[...,y:y+48,x:x+48],lms[...,y:y+48,x:x+48],ms[...,y//4:(y+48)//4,x//4:(x+48)//4])
                differences.append((full[...,y+16:y+32,x+16:x+32]-tile[...,16:32,16:32]).abs().mean().item())
        row['overlapping_tiles_interior_mae']=differences
        # Block-level locality isolates the attention from backbone interpolation.
        block=model.SSA_blocks[0].eval()
        p,m=torch.rand(1,1,64,64,device=device),torch.rand(1,5,64,64,device=device)
        with torch.no_grad():
            full=block(p,m); crop=block(p[...,16:48,16:48],m[...,16:48,16:48])
        error=(full[...,24:40,24:40]-crop[...,8:24,8:24]).abs().max().item()
        row['ssa_block_crop_interior_max_abs']=error
        if variant in ('A2','A3'): assert error < 2e-6
        model.train()
        pan = torch.rand(2,1,32,48,device=device)
        lms = torch.rand(2,4,32,48,device=device)
        ms = torch.rand(2,4,8,12,device=device)
        model(pan,lms,ms).square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if device.startswith('cuda'):
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                amp_pred = model(pan,lms,ms)
                amp_loss = amp_pred.float().square().mean()
            torch.amp.GradScaler('cuda',init_scale=16).scale(amp_loss).backward()
            assert torch.isfinite(amp_pred).all()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        block = model.SSA_blocks[0].eval()
        if variant == 'A3':
            # Compare chunked computation against a single full-window unfolding.
            block.chunk_rows = 3
            p,m = torch.rand(1,1,12,20,device=device),torch.rand(1,5,12,20,device=device)
            with torch.no_grad():
                chunk = block(p,m); block.chunk_rows = 100
                assert torch.allclose(chunk,block(p,m),atol=1e-6)
        if variant == 'A3':
            block.chunk_rows=3
            with torch.no_grad():
                actual,weights=block.attend(p,m,True)
                key,query=block.features(p,m); value=block.value(key)
                for y,x in ((0,0),(5,7),(11,19)):
                    neighbors=[(yy,xx) for yy in range(max(0,y-2),min(12,y+3)) for xx in range(max(0,x-2),min(20,x+3))]
                    ks=torch.stack([key[0,:,yy,xx] for yy,xx in neighbors])
                    vs=torch.stack([value[0,:,yy,xx] for yy,xx in neighbors])
                    attention=(ks.float()@query[0,:,y,x].float()/6**0.5).softmax(0)
                    expected=(attention[:,None]*vs.float()).sum(0)
                    assert torch.allclose(actual[0,:,y,x].float(),expected,atol=2e-6)
        size_rows = []
        for size in (64,256,512):
            with torch.no_grad():
                # Constant projected features isolate normalization from content.
                for name in ('conv1t6','conv6t6','conv7t6_3'):
                    conv = getattr(block,name); conv.weight.zero_(); conv.bias.fill_(1)
                p = torch.ones(1,1,size,size,device=device)
                m = torch.ones(1,5,size,size,device=device)
                if variant == 'A3':
                    out,weights = block.attend(p,m,True)
                    assert torch.allclose(weights.sum(1),torch.ones_like(weights[:,0]),atol=1e-6)
                    assert (weights[:, :12, 0,0] == 0).all()
                else:
                    out = block(p,m)
                    weights = block.attention(p,m)[1] if variant != 'A0' else out
                size_rows.append(dict(size=size, center_output=out[0,:,size//2,size//2].mean().item(), weight_mean=weights.mean().item()))
        row['constant_feature_size_probe'] = size_rows
        rows[variant] = row
        del model
    with tempfile.TemporaryDirectory(prefix='ssa-fusion-') as directory:
        folder = Path(directory)
        script = ROOT/'scripts/kaggle_ssa_fusion_cell.py'
        notebook = json.loads(script.with_suffix('.ipynb').read_text()) if script.with_suffix('.ipynb').exists() else json.loads((ROOT/'scripts/kaggle_ssa_fusion.ipynb').read_text())
        assert len(notebook['cells']) == 1 and ''.join(notebook['cells'][0]['source']) == script.read_text()
        worker_path = folder/'worker.py'; worker_path.write_text(embedded_worker(script))
        spec = importlib.util.spec_from_file_location('fusion_worker_test',worker_path)
        worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
        rng = np.random.default_rng(46)
        for split,n in (('train',7),('val',3)):
            for key,c,h in (('gt',4,16),('lms',4,16),('pan',1,16),('ms',4,4)):
                np.save(folder/f'{split}_{key}.npy',rng.random((n,c,h,h),dtype=np.float32))
        resume = {}
        for variant in ('A0','A1','A2','A3'):
            cfg = dict(cache=str(folder),output=str(folder/'continuous'),code_root=str(ROOT.parent),seed=46,deterministic=True,k=6,lr=1e-4,amp=False,micro_batch=2,batch_size=4,epochs=1,epochs_this_session=1,workers=0,checkpoint_steps=2,deadline=time.time()+1200,fingerprint='synthetic-test',gpu_resident=False)
            worker.train(cfg,variant,'cpu')
            cfg['output'] = str(folder/'resumed'); cfg['test_stop_batch'] = 2
            worker.train(cfg,variant,'cpu')
            partial = worker.load_checkpoint(folder/'resumed'/variant)
            assert partial['next_batch'] == 2 and partial['epoch'] == 0
            cfg.pop('test_stop_batch'); worker.train(cfg,variant,'cpu')
            a = worker.load_checkpoint(folder/'continuous'/variant)
            b = worker.load_checkpoint(folder/'resumed'/variant)
            assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
            assert a['history'][0]['val_mse'] == b['history'][0]['val_mse']
            cfg['fingerprint'] = 'different'
            try: worker.train(cfg,variant,'cpu')
            except ValueError: pass
            else: raise AssertionError('Mismatched configuration accepted')
            (folder/'resumed'/variant/'latest.pt').write_bytes(b'corrupt')
            assert worker.load_checkpoint(folder/'resumed'/variant) is not None
            resume[variant] = dict(mid_epoch_exact=True, mismatch_rejected=True, corrupted_latest_fallback=True)
    result = dict(scope='synthetic engineering checks; no research performance claim', device=device,torch=str(torch.__version__),models=rows,resume=resume,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/ssamrn/models/ssa_fusion.py',ROOT/'scripts/kaggle_ssa_fusion_cell.py')})
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+'\n')
    print('Verified:', output)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cpu'); p.add_argument('--output',type=Path,default=ROOT/'docs/assets/ssa_fusion_preflight/verification.json')
    args=p.parse_args(); verify(args.output,args.device)
