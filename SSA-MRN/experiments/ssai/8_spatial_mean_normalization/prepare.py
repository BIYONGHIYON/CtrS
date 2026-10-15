"""Verify control assets, measure diagnostics, and freeze a fresh experiment8 plan."""
import argparse, ast, importlib, json, math, shutil, sys, time, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common import ssai_suite as suite
import h5py, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from common.model import RestoredPansharpeningNet
module = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.model')
diag = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.diagnostics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--detail-runtime', type=Path, required=True)
    a = parser.parse_args()
    if a.runtime.exists(): raise FileExistsError('Fresh runtime required')
    if shutil.disk_usage(a.previous).free < 20*1024**3: raise RuntimeError('At least20GiB free required')
    print('phase: baseline environment/source/cache hashes',flush=True)
    old = json.loads((a.previous/'plan.json').read_text())
    actual = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)}
    for k, v in actual.items():
        if old['environment'][k] != v: raise ValueError('Environment mismatch: '+k)
    for name in ('common/model.py', 'references/upstream/network.py'):
        if suite.sha(ROOT/name) != old['source_hashes'][name]: raise ValueError('Baseline source changed: '+name)
    # Reuse only numerical functions proven identical to the real experiment6/7 source.
    previous_suite = a.detail_runtime/'provenance/source/common/ssai_suite.py'
    def syntax(path, name):
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name==name)
        return ast.dump(node, include_attributes=False)
    names = ('configure', 'run_epoch', 'validate', 'Cached', 'GPUCachedLoader', 'PartialGPUCachedLoader', 'make_loader', 'save')
    for name in names:
        if syntax(ROOT/'common/ssai_suite.py', name) != syntax(previous_suite, name): raise ValueError('Shared numerical path changed: '+name)
    cache = Path(old.get('cache_root', a.previous/'cache'))
    for split in ('train', 'val'):
        info = old['data'][split]
        if suite.sha(info['path']) != info['sha256']: raise ValueError('Original data changed')
        for key, sha in info['cache'].items():
            if suite.sha(cache/split/(key+'.npy')) != sha: raise ValueError('Cache changed')
    print('phase: control checkpoints',flush=True)
    refs = {}
    control_best = control_latest = None
    for arm in ('A0', 'A2'):
        complete = json.loads((a.previous/arm/'complete.json').read_text())
        refs[arm] = {}
        for kind in ('best', 'latest'):
            path = a.previous/arm/(kind+'.pt'); sha = suite.sha(path)
            if sha != complete[kind+'_sha256']: raise ValueError('Reference weight hash mismatch')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            for key in ('seed','sensor','k','epochs','effective_batch','micro_batch','lr','precision','amp','deterministic','tf32','data'):
                if ck['config']['plan'][key] != old[key]: raise ValueError('Control condition mismatch: '+key)
            suite.model(arm).load_state_dict(ck['model'], strict=True)
            refs[arm][kind] = {'path': str(path), 'sha256': sha, 'epoch': ck['epoch']}
            if arm=='A0' and kind=='best': control_best = ck
            if arm=='A0' and kind=='latest': control_latest = ck
    print('phase: model preflight',flush=True)
    a.runtime.mkdir(parents=True)
    suite.configure(42, 'cuda')
    control = RestoredPansharpeningNet(4,4)
    initial = {k: v.clone() for k, v in control.state_dict().items()}
    init_hash = diag.state_hash(initial)
    # Reproduce A0 via its actual training suite, not merely its seed label.
    spec = importlib.util.spec_from_file_location('actual_experiment4_suite', a.previous/'provenance/source/common/ssai_suite.py')
    actual_suite = importlib.util.module_from_spec(spec); spec.loader.exec_module(actual_suite)
    actual_suite.configure(42, 'cuda')
    initial_original = actual_suite.model('A0').state_dict()
    assert diag.state_hash(initial_original)==init_hash
    history = json.loads((a.previous/'A0/history.json').read_text())
    first_mse = next(row['val_mse'] for row in history if row['epoch']==1)
    assert control_latest['epoch']==100
    checks = {'passed': False, 'original_initial_hash': init_hash, 'actual_experiment4_initial_hash': diag.state_hash(initial_original), 'models': {}}
    va = suite.Cached(cache/'val')
    validation_indices = list(range(512))
    x = [v.unsqueeze(0).cuda() for v in va[0][:3]]
    control.load_state_dict(control_best['model'], strict=True); control.cuda().eval()
    with torch.inference_mode(): expected = control(*x)
    for arm in ('N1', 'N2'):
        suite.configure(42, 'cuda')
        net = module.MeanNormalizedNet(learn_gamma=arm=='N2').cuda()
        common = {k:v for k,v in net.state_dict().items() if not k.endswith('.gamma')}
        assert diag.state_hash(common)==init_hash
        assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in control.parameters())+(12 if arm=='N2' else 0)
        gamma_gradients = None
        if arm=='N2':
            torch.nn.functional.mse_loss(net(*x), va[0][3].unsqueeze(0).cuda()).backward()
            gamma_gradients = {n:float(v.grad) for n,v in net.named_parameters() if n.endswith('.gamma')}
            assert len(gamma_gradients)==12 and all(math.isfinite(v) and v!=0 for v in gamma_gradients.values())
            net.zero_grad(set_to_none=True)
        forwards = []
        with torch.inference_mode():
            for h,w in ((64,64),(256,256),(512,512),(32,48)):
                inputs = [torch.rand(1,1,h,w,device='cuda'), torch.rand(1,4,h,w,device='cuda'), torch.rand(1,4,h//4,w//4,device='cuda')]
                output = net(*inputs)
                assert output.shape==(1,4,h,w) and torch.isfinite(output).all()
                max_error = 0.
                for stage, scale in zip(diag.STAGES,(1,2,4)):
                    pan = torch.rand(1,1,h//scale,w//scale,device='cuda')
                    feature = torch.rand(1,5,h//scale,w//scale,device='cuda')
                    for block in getattr(net,stage):
                        _, weight = block.components(pan,feature)
                        max_error=max(max_error,float((weight.mean((-1,-2))-1).abs().max()))
                assert max_error<=1e-6
                forwards.append({'shape':[h,w], 'finite':True, 'weight_mean_max_abs_error':max_error})
            for stage in diag.STAGES:
                for block in getattr(net,stage):
                    block.normalize_mean=False
                    if isinstance(block.gamma, torch.Tensor): block.gamma.fill_(1.)
            state = net.state_dict()
            state.update(control_best['model'])
            net.load_state_dict(state,strict=True)
            restored = net(*x)
            assert torch.equal(restored,expected), (restored-expected).abs().max().item()
            rectangular = [torch.rand(1,1,32,48,device='cuda'),torch.rand(1,4,32,48,device='cuda'),torch.rand(1,4,8,12,device='cuda')]
            assert torch.equal(net(*rectangular), control(*rectangular))
        checks['models'][arm]={'common_initial_hash':init_hash,'added_parameters':12 if arm=='N2' else 0,'gamma_gradients':gamma_gradients,'forwards':forwards,'original_best_exact_restoration_64_and_rectangle':True}
        del net
    print('phase: baseline activation/gradient diagnostics',flush=True)
    baseline = {'a0_best':refs['A0']['best'], 'a0_latest':refs['A0']['latest'], 'initial_sha256':init_hash, 'validation_indices':validation_indices, 'A0_epoch1_validation_mse':first_mse}
    batches = DataLoader(Subset(va,validation_indices),batch_size=4,shuffle=False,num_workers=0)
    baseline['validation_512'] = diag.rms_ratios(control,batches,'cuda')
    for protocol,filename in [('RR','test_qb_multiExm1.h5'),('FR','test_qb_OrigScale_multiExm1.h5')]:
        paths=list(Path(old['data_root']).rglob(filename));assert len(paths)==1
        with h5py.File(paths[0],'r')as f:
            inputs=[torch.from_numpy(np.asarray(f[k][0],dtype=np.float32)/2047.).unsqueeze(0) for k in ('pan','lms','ms')]
        assert tuple(inputs[0].shape[-2:])==((256,256)if protocol=='RR'else(512,512))
        baseline[protocol+'_scene0']={'index':0,'data_sha256':suite.sha(paths[0]),'rms':diag.rms_ratios(control,[inputs],'cuda')}
    control.load_state_dict(control_latest['model'],strict=True)
    baseline['A0_latest_weight_relative_change']=diag.weight_changes(control,initial)
    control.load_state_dict(initial,strict=True)
    g=torch.Generator().manual_seed(42);torch.empty((),dtype=torch.int64).random_(generator=g)
    indices=torch.randperm(17139,generator=g)[:640].tolist()
    baseline['train_gradient_indices']=indices
    batches=DataLoader(Subset(suite.Cached(cache/'train'),indices),batch_size=32,shuffle=False,num_workers=0)
    baseline['initial_gradients']=diag.gradient_groups(control,batches,'cuda')
    assert diag.state_hash(control.state_dict())==init_hash
    # Same twenty batches, no optimizer update: test whether normalization actually lifts gradients.
    baseline['normalized_initial_gradients']={}
    for arm in ('N1','N2'):
        suite.configure(42,'cuda');net=module.MeanNormalizedNet(learn_gamma=arm=='N2').cuda()
        baseline['normalized_initial_gradients'][arm]=diag.gradient_groups(net,batches,'cuda')
        del net
    print('phase: freeze source and plan',flush=True)
    checks['passed']=True
    checks['cache_verification']={'method':'reuse verified two full epochs of exact FP32 batches/RNG and three exact Adam steps from experiments6/7','same_numerical_functions':list(names),'source_sha256':suite.sha(previous_suite),'evidence_sha256':suite.sha(ROOT/'experiments/ssai/6_ms_conditioned_detail/results/partial_cache_check.json')}
    suite.write(a.runtime/'preflight.json',checks)
    suite.write(a.runtime/'diagnostics.json',baseline)
    p={k:v for k,v in old.items() if k not in ('initial_checkpoints','optimization_benchmark','vram_safety','reference_area_candidate')}
    p.update(arms=['N1','N2'],cache_mode='gpu_partial',cache_fraction=.85,cache_root=str(cache),reference_runs=refs,environment=actual,created_utc=time.time(),initial_common_sha256=init_hash,diagnostic_validation_indices=validation_indices,divergence={'first_epoch_mse_limit':2*first_mse,'A0_epoch1_validation_mse':first_mse,'nonfinite_loss_or_gradient':'stop before next update','automatic_restart':False},learning_success={'epoch':100,'convolution_weights_only':True,'all_stages_min_relative_change':.10,'all_stages_min_ratio_to_A0_latest':2.0,'gamma_excluded':True},followup={'experiment':9,'requires_remote_preservation':True,'requires_selected_B_learning_success':True,'automatic_launch_in_this_controller':False})
    paths=list((ROOT/'common').glob('*.py'))+[ROOT/'references/upstream/network.py',ROOT/'experiments/ssai/3_calibrated_sigmoid_gating/model.py']
    paths += list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'config.json']
    p['source_hashes']={path.relative_to(ROOT).as_posix():suite.sha(path) for path in paths}
    for path in paths:
        dest=a.runtime/'provenance/source'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    suite.write(a.runtime/'plan.json',p);suite.write(a.runtime/'provenance/plan.json',p)
    with zipfile.ZipFile(a.runtime/'provenance/training_source.zip','w',compression=zipfile.ZIP_DEFLATED)as z:
        for path in (a.runtime/'provenance/source').rglob('*'):
            if path.is_file():z.write(path,path.relative_to(a.runtime/'provenance'))
        z.write(a.runtime/'provenance/plan.json','plan.json')
    print(json.dumps({'prepared':True,'preflight_passed':True,'initial_sha256':init_hash,'first_epoch_mse_limit':2*first_mse,'diagnostics':str(a.runtime/'diagnostics.json')}),flush=True)


if __name__=='__main__': main()
