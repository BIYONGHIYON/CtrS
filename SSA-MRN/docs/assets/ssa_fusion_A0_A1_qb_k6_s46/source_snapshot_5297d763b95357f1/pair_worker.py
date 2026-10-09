import os, sys, json, math, time, hashlib, signal, argparse, shutil
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

STOP = False

def stop_signal(*args):
    global STOP
    STOP = True


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def atomic_checkpoint(value, path):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(cpu_tree(value), tmp)
    with open(tmp, 'rb') as f: os.fsync(f.fileno())
    if path.exists(): os.replace(path, path.with_suffix('.previous.pt'))
    os.replace(tmp, path)


def load_checkpoint(folder):
    for name in ('latest.pt', 'latest.previous.pt'):
        path = folder / name
        if path.exists():
            try: return torch.load(path, map_location='cpu', weights_only=True)
            except Exception as exc: print(f'{name} 읽기 실패: {exc}', flush=True)
    if any(folder.glob('latest*pt')): raise RuntimeError('복원 가능한 latest 가중치가 없습니다.')
    return None


def epoch_order(length, seed, epoch):
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch * 100003)).tolist()












class CachedPairs(Dataset):
    def __init__(self, cache, split, variant, seed):
        self.base = {k: np.load(cache / f'{split}_{k}.npy', mmap_mode='r') for k in ('gt','pan','lms','ms')}
        self.variant, self.seed, self.epoch = variant, seed, 0
        self.deltas = {}
    def __len__(self): return len(self.base['gt'])
    def __getitem__(self, token):
        index, epoch = token if isinstance(token, tuple) else (token, 0)
        sample = {k: np.array(v[index], copy=True) for k, v in self.base.items()}
        return {k: torch.from_numpy(v) for k,v in sample.items()}


class ResidentPairs:
    """Batch gathers and augmentation on device; no per-sample CPU copies or H2D."""
    def __init__(self, source, device):
        self.device, self.variant, self.seed = device, source.variant, source.seed
        def upload(array):
            out = torch.empty(array.shape, dtype=torch.float32, device=device)
            for start in range(0, len(array), 256):
                chunk = torch.from_numpy(np.array(array[start:start+256], copy=True))
                out[start:start+len(chunk)].copy_(chunk)
            return out
        self.base = {k: upload(v) for k,v in source.base.items()}
        self.deltas = {i: {k: upload(v) for k,v in values.items()} for i,values in source.deltas.items()}
    def __len__(self): return len(self.base['gt'])
    def batch(self, indices, epoch=0):
        ids = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        sample = {k: v.index_select(0,ids) for k,v in self.base.items()}
        return sample


class ResidentLoader:
    def __init__(self, dataset, size, sampler=None):
        self.dataset,self.size,self.sampler = dataset,size,sampler
    def __iter__(self):
        if self.sampler is not None:
            for tokens in self.sampler:
                indices = [t[0] for t in tokens]
                yield self.dataset.batch(indices, self.sampler.epoch)
        else:
            for start in range(0,len(self.dataset),self.size):
                yield self.dataset.batch(range(start,min(start+self.size,len(self.dataset))))
    def __len__(self):
        return len(self.sampler) if self.sampler is not None else math.ceil(len(self.dataset)/self.size)


def resident_bytes(*datasets):
    return sum(v.nbytes for d in datasets for v in d.base.values()) + sum(v.nbytes for d in datasets for bank in d.deltas.values() for v in bank.values())


class ResumeBatches:
    def __init__(self, dataset, size, seed):
        self.dataset, self.size, self.seed = dataset, size, seed
        self.epoch, self.start = 0, 0
    def __iter__(self):
        order = epoch_order(len(self.dataset), self.seed, self.epoch)
        for start in range(self.start * self.size, len(order), self.size):
            yield [(i, self.epoch) for i in order[start:start+self.size]]
    def __len__(self): return math.ceil(len(self.dataset)/self.size) - self.start


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    stats = torch.zeros(4, device=device, dtype=torch.float64)
    images = 0
    for b in loader:
        pan, lms, ms, gt = [b[k].to(device, non_blocking=True) for k in ('pan','lms','ms','gt')]
        # All validation metrics computed from FP32 inference, regardless of AMP training.
        pred = model(pan, lms, ms)
        difference = pred - gt
        stats[0] += difference.square().sum(); stats[1] += gt.numel()
        stats[2] += (-10 * difference.square().mean((-2,-1)).clamp_min(1e-30).log10()).mean(1).sum()
        norms = pred.norm(dim=1) * gt.norm(dim=1)
        valid = norms > 0
        angles = torch.rad2deg(torch.acos(((pred * gt).sum(1) / norms.clamp_min(1e-30)).clamp(-1,1)))
        counts = valid.flatten(1).sum(1)
        if (counts == 0).any(): raise ValueError('SAM 계산 불가')
        stats[3] += ((angles * valid).flatten(1).sum(1) / counts).sum()
        images += len(gt)
    s = stats.cpu().tolist()
    if not all(math.isfinite(x) for x in s): raise RuntimeError('validation nonfinite')
    return s[0]/s[1], s[2]/images, s[3]/images


def train(config, variant, device='cuda:0'):
    from ssamrn.models.ssa_fusion import FusionPansharpeningNet
    folder = Path(config['output']) / variant
    folder.mkdir(parents=True, exist_ok=True)
    cache = Path(config['cache'])
    torch.set_num_threads(2)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if device.startswith('cuda'): torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = not config['deterministic']
    torch.use_deterministic_algorithms(config['deterministic'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    epoch, next_batch, total, count, best, history, skipped = 0, 0, 0., 0, float('inf'), [], 0
    started = time.time()
    def report(**updates):
        record = dict(variant=variant, stage='학습', epoch=epoch+1, target=config['epochs'],
                      elapsed_seconds=time.time()-started, updated=time.time())
        record.update(updates)
        atomic_json(record, folder / 'status.json')
    report(stage='준비', percent=0)
    tr = CachedPairs(cache, 'train', variant, config['seed'])
    va = CachedPairs(cache, 'val', 'baseline', config['seed'])
    sampler = ResumeBatches(tr, config['micro_batch'], config['seed'])
    use_resident = False
    if config.get('gpu_resident',True) and device.startswith('cuda'):
        needed = resident_bytes(tr,va)
        free,total_memory = torch.cuda.mem_get_info()
        # Keep 3 GiB for model, activations, cuDNN workspaces and checkpoint operations.
        use_resident = needed + (3 << 30) < free
        print(f'GPU cache: need {needed/(1<<30):.2f} GiB; free {free/(1<<30):.2f} GiB; enabled={use_resident}',flush=True)
        if use_resident:
            report(stage='GPU 데이터 준비', percent=0, gpu_cache_gib=needed/(1<<30))
            tr_gpu,va_gpu = ResidentPairs(tr,device),ResidentPairs(va,device)
            loader = ResidentLoader(tr_gpu,config['micro_batch'],sampler)
            validation = ResidentLoader(va_gpu,config['micro_batch'])
    if not use_resident:
        options = dict(num_workers=config['workers'], pin_memory=device.startswith('cuda'))
        if config['workers']: options.update(persistent_workers=True, prefetch_factor=2)
        loader = DataLoader(tr, batch_sampler=sampler, **options)
        validation = DataLoader(va, batch_size=config['micro_batch'], shuffle=False, **options)
    model = FusionPansharpeningNet(4, config['k'], variant).to(device)
    initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    provenance = dict(parameters_total=sum(p.numel() for p in model.parameters()), parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), config=config, initial_model_sha256=initial_hash, torch=str(torch.__version__),
                      gpu=torch.cuda.get_device_name(0) if device.startswith('cuda') else 'CPU')
    if not (folder/'provenance.json').exists(): atomic_json(provenance, folder/'provenance.json')
    atomic_json(provenance, folder / ('session_' + time.strftime('%Y%m%d_%H%M%S') + '.json'))
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    amp = config['amp'] and device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=1024.)
    ck = load_checkpoint(folder)
    if ck:
        if ck['fingerprint'] not in config.get('accepted_fingerprints',[config['fingerprint']]) or ck['variant'] != variant:
            raise ValueError('코드/데이터/학습 조건이 달라 재개할 수 없습니다.')
        if ck['fingerprint'] != config['fingerprint']:
            atomic_json(dict(previous_fingerprint=ck['fingerprint'], current_fingerprint=config['fingerprint'], policy='audited GPU-cache-only implementation migration'),folder/'resume_migration.json')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        epoch, next_batch = ck['epoch'], ck['next_batch']
        total, count, best, history, skipped = ck['total'], ck['count'], ck['best_mse'], ck['history'], ck['skipped_steps']
        torch.set_rng_state(ck['rng']); random.setstate(ck['python_rng']); np.random.set_state(('MT19937', np.asarray(ck['numpy_rng'][1], dtype=np.uint32), *ck['numpy_rng'][2:]))
        if amp or device.startswith('cuda'): torch.cuda.set_rng_state(ck['cuda_rng'], 0)
        print(f'{variant}: epoch {epoch+1}, batch {next_batch}부터 복원', flush=True)
    total = torch.tensor(total,dtype=torch.float64,device=device)
    stepped = [False]
    step_hook = opt.register_step_post_hook(lambda optimizer,args,kwargs: stepped.__setitem__(0,True))
    def checkpoint(name='latest.pt'):
        value = dict(config=dict(config, sensor='QB', variant=variant, training_variant=variant), fingerprint=config['fingerprint'], variant=variant, epoch=epoch, next_batch=next_batch,
                     best_mse=best, total=total.item(), count=count, history=history, skipped_steps=skipped,
                     model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                     python_rng=random.getstate(), numpy_rng=(np.random.get_state()[0], np.random.get_state()[1].tolist(), *np.random.get_state()[2:]), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0) if device.startswith('cuda') else torch.empty(0,dtype=torch.uint8))
        atomic_checkpoint(value, folder/name)
        atomic_json(history, folder/'history.json')
        (folder/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        return value
    stop_epoch = min(config['epochs'], epoch + config['epochs_this_session'])
    batches = math.ceil(len(tr)/config['micro_batch'])
    accumulation = config['batch_size']//config['micro_batch']
    last_save = time.time()
    while epoch < stop_epoch:
        model.train()
        sampler.epoch, sampler.start = epoch, next_batch
        tick = time.time()
        opt.zero_grad(set_to_none=True)
        for local_batch, b in enumerate(loader):
            batch = sampler.start + local_batch
            group_start = (batch//accumulation)*accumulation
            # Account for the smaller final accumulation group without changing effective weighting.
            group_samples = min(config['batch_size'], len(tr) - group_start*config['micro_batch'])
            pan, lms, ms, gt = [b[k].to(device, non_blocking=True) for k in ('pan','lms','ms','gt')]
            with torch.autocast(device_type=device.split(':')[0], dtype=torch.float16, enabled=amp):
                pred = model(pan,lms,ms)
                loss = F.mse_loss(pred.float(), gt.float())
            if not torch.isfinite(loss): raise RuntimeError('학습 loss nonfinite; 출력 보존 후 원인 확인 필요')
            scaler.scale(loss * (len(gt)/group_samples)).backward()
            total += loss.detach().double()*len(gt); count += len(gt)
            boundary = (batch+1)%accumulation == 0 or batch+1 == batches
            if boundary:
                stepped[0] = False
                scaler.step(opt); scaler.update()
                skipped += int(not stepped[0])
                opt.zero_grad(set_to_none=True)
                next_batch = batch+1
                if next_batch % config['checkpoint_steps'] == 0 or time.time()-last_save > 300:
                    checkpoint(); last_save = time.time()
                if STOP or time.time() >= config['deadline'] or (config.get('test_stop_batch') and next_batch >= config['test_stop_batch']):
                    checkpoint()
                    report(stage='일시정지 · 저장 완료', percent=100*next_batch/batches, batch=next_batch, batches=batches)
                    return
            if boundary and batch%10 < accumulation:
                rate = (local_batch+1)/(time.time()-tick)
                report(percent=100*(batch+1)/batches, batch=batch+1, batches=batches,
                       train_mse=total.item()/count, eta_epoch_seconds=(batches-batch-1)/rate,
                       best_val_mse=best if math.isfinite(best) else None,
                       samples_per_second=rate*config['micro_batch'], gpu_resident=use_resident,
                       gpu_allocated_gib=torch.cuda.memory_allocated()/(1<<30) if device.startswith('cuda') else 0.)
        report(stage='검증', percent=100, batch=batches, batches=batches)
        val, psnr, sam = validate(model, validation, device)
        improved = val < best
        best = min(best, val)
        row = dict(epoch=epoch+1, train_mse=total.item()/count, val_mse=val, val_psnr_band_mean_peak1=psnr,
                   val_sam_deg=sam, seconds_this_segment=time.time()-tick, skipped_amp_steps=skipped)
        history.append(row)
        print(json.dumps(row), flush=True)
        epoch += 1; next_batch = 0; total.zero_(); count = 0
        if improved: checkpoint('best.pt')
        checkpoint(); last_save = time.time()
        if STOP or time.time() >= config['deadline']: break
    report(stage='완료' if epoch >= config['epochs'] else '세션 분량 완료 · 재개 가능', percent=100, completed_epochs=epoch,
           best_val_mse=best, next_batch=next_batch)
    if epoch >= config['epochs']:
        atomic_json(dict(epochs=epoch, best_val_mse=best, fingerprint=config['fingerprint']), folder/'complete.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--variant', choices=['A0','A1','A2','A3'], required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    sys.path.insert(0, str(Path(config['code_root'])/'SSA-MRN/src'))
    signal.signal(signal.SIGTERM, stop_signal); signal.signal(signal.SIGINT, stop_signal)
    try: train(config, args.variant)
    except Exception as exc:
        folder = Path(config['output']) / args.variant
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(dict(stage='실패', error=str(exc), updated=time.time()), folder/'status.json')
        raise

if __name__ == '__main__': main()
