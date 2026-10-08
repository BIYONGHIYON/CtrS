import os, sys, json, math, time, hashlib, signal, argparse, shutil
from pathlib import Path
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


def bank_choice(index, seed, epoch):
    # Worker count and prefetch do not affect augmentation or shuffling on resume.
    return ((int(index) * 2654435761 + seed * 2246822519 + epoch * 3266489917) & 0xffffffff) % 3


def kernels_for_factor(factor):
    from ssamrn.observation import GNYQ, mtf_kernels
    original = GNYQ['QB']
    try:
        GNYQ['QB'] = [v * factor for v in original]
        return torch.from_numpy(mtf_kernels('QB')).float()
    finally:
        GNYQ['QB'] = original


@torch.no_grad()
def observe(gt, kernel, phases):
    # Exact source padding and phase; stride-4 convolution avoids full-HR output.
    padded = F.pad(gt, (20, 20, 20, 20), mode='replicate')
    out = gt.new_empty(len(gt), 4, gt.shape[-2] // 4, gt.shape[-1] // 4)
    for r, c in ((2,2), (1,2), (2,1)):
        selected = (phases[:, 0] == r) & (phases[:, 1] == c)
        if selected.any():
            out[selected] = F.conv2d(padded[selected, :, r:, c:], kernel, stride=4, groups=4)
    return out


@torch.no_grad()
def make_delta(gt, kernel_difference, phases):
    delta = observe(gt, kernel_difference, phases)
    # 41-tap filter needs a 20-HR-pixel halo. Only the audited 6x6 LR interior changes.
    mask = torch.zeros_like(delta)
    mask[..., 5:-5, 5:-5] = 1
    return delta * mask


def build_banks(cache, phases, config, device, report):
    from ssamrn.models.interp23 import interp23tap
    stamp = cache / 'mtf_bank_complete.json'
    if stamp.exists():
        recorded = json.loads(stamp.read_text())
        if recorded['fingerprint'] != config['fingerprint']: raise ValueError('MTF 캐시 설정 불일치')
        return recorded
    gt = np.load(cache / 'train_gt.npy', mmap_mode='r')
    ms = np.load(cache / 'train_ms.npy', mmap_mode='r')
    nominal = kernels_for_factor(1.).to(device)
    differences = [kernels_for_factor(f).to(device) - nominal for f in config['mtf_factors']]
    banks = []
    for i in (1, 2):
        dm = np.lib.format.open_memmap(cache / f'delta_ms_{i}.npy', mode='w+', dtype='float32', shape=ms.shape)
        dl = np.lib.format.open_memmap(cache / f'delta_lms_{i}.npy', mode='w+', dtype='float32', shape=gt.shape)
        banks.append((dm, dl))
    rmse_sum, rmse_max, count = 0., 0., 0
    for start in range(0, len(gt), 128):
        end = min(start + 128, len(gt))
        g = torch.from_numpy(np.array(gt[start:end])).to(device)
        m = torch.from_numpy(np.array(ms[start:end])).to(device)
        p = torch.tensor(phases[start:end], device=device)
        errors = (observe(g, nominal, p)[..., 5:-5, 5:-5] - m[..., 5:-5, 5:-5]) * 2047
        rmses = errors.square().mean((1,2,3)).sqrt()
        rmse_sum += rmses.sum().item(); rmse_max = max(rmse_max, rmses.max().item()); count += len(g)
        if not math.isfinite(rmse_max) or rmse_max > 2:
            raise ValueError('원본 TRAIN의 MTF/위상 검증 실패: 2 DN 초과')
        for bank, difference in zip(banks, differences):
            d = make_delta(g, difference, p)
            l = interp23tap(d)
            bank[0][start:end] = d.cpu().numpy()
            bank[1][start:end] = l.cpu().numpy()
        report(stage='MTF 캐시', percent=100 * end / len(gt), batch=0, batches=0)
        if STOP: raise InterruptedError('MTF 캐시 준비 중 종료 요청')
        if time.time() > config['deadline']: raise TimeoutError('MTF 준비 중 세션 시간 제한')
    for dm, dl in banks: dm.flush(); dl.flush()
    record = dict(fingerprint=config['fingerprint'], samples=count, mean_rmse_dn=rmse_sum/count,
                  max_rmse_dn=rmse_max, factors=config['mtf_factors'], interior_lr_margin=5,
                  policy='MS + (D_jitter-D_nominal)(GT) in audited interior; LMS + interp23(delta_MS)')
    atomic_json(record, stamp)
    return record


class CachedPairs(Dataset):
    def __init__(self, cache, split, variant, seed):
        self.base = {k: np.load(cache / f'{split}_{k}.npy', mmap_mode='r') for k in ('gt','pan','lms','ms')}
        self.variant, self.seed, self.epoch = variant, seed, 0
        self.deltas = {}
        if variant == 'mtf_aug':
            self.deltas = {i: {k: np.load(cache / f'delta_{k}_{i}.npy', mmap_mode='r') for k in ('lms','ms')} for i in (1,2)}
    def __len__(self): return len(self.base['gt'])
    def __getitem__(self, token):
        index, epoch = token if isinstance(token, tuple) else (token, 0)
        sample = {k: np.array(v[index], copy=True) for k, v in self.base.items()}
        bank = bank_choice(index, self.seed, epoch) if self.variant == 'mtf_aug' else 0
        if bank:
            for k in ('lms', 'ms'): sample[k] += self.deltas[bank][k][index]
        return {k: torch.from_numpy(v) for k,v in sample.items()}


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
    from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
    folder = Path(config['output']) / variant
    folder.mkdir(parents=True, exist_ok=True)
    cache = Path(config['cache'])
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
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
    if variant == 'mtf_aug':
        profile = json.loads((Path(config['code_root']) / 'qb_profile.json').read_text())
        audit = build_banks(cache, profile['phases'], config, device, report)
        atomic_json(audit, folder / 'augmentation_audit.json')
    tr = CachedPairs(cache, 'train', variant, config['seed'])
    va = CachedPairs(cache, 'val', 'baseline', config['seed'])
    sampler = ResumeBatches(tr, config['micro_batch'], config['seed'])
    options = dict(num_workers=config['workers'], pin_memory=device.startswith('cuda'))
    if config['workers']: options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(tr, batch_sampler=sampler, **options)
    validation = DataLoader(va, batch_size=config['micro_batch'], shuffle=False, **options)
    model = RestoredPansharpeningNet(4, config['k']).to(device)
    initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    provenance = dict(config=config, initial_model_sha256=initial_hash, torch=str(torch.__version__),
                      gpu=torch.cuda.get_device_name(0) if device.startswith('cuda') else 'CPU')
    if not (folder/'provenance.json').exists(): atomic_json(provenance, folder/'provenance.json')
    atomic_json(provenance, folder / ('session_' + time.strftime('%Y%m%d_%H%M%S') + '.json'))
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    amp = config['amp'] and device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=1024.)
    ck = load_checkpoint(folder)
    if ck:
        if ck['fingerprint'] != config['fingerprint'] or ck['variant'] != variant:
            raise ValueError('코드/데이터/학습 조건이 달라 재개할 수 없습니다.')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        epoch, next_batch = ck['epoch'], ck['next_batch']
        total, count, best, history, skipped = ck['total'], ck['count'], ck['best_mse'], ck['history'], ck['skipped_steps']
        torch.set_rng_state(ck['rng'])
        if amp or device.startswith('cuda'): torch.cuda.set_rng_state(ck['cuda_rng'], 0)
        print(f'{variant}: epoch {epoch+1}, batch {next_batch}부터 복원', flush=True)
    def checkpoint(name='latest.pt'):
        value = dict(config=dict(config, sensor='QB', variant='baseline', training_variant=variant), fingerprint=config['fingerprint'], variant=variant, epoch=epoch, next_batch=next_batch,
                     best_mse=best, total=total, count=count, history=history, skipped_steps=skipped,
                     model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0) if device.startswith('cuda') else torch.empty(0,dtype=torch.uint8))
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
            total += loss.detach().item()*len(gt); count += len(gt)
            boundary = (batch+1)%accumulation == 0 or batch+1 == batches
            if boundary:
                old_scale = scaler.get_scale()
                scaler.step(opt); scaler.update()
                skipped += int(scaler.get_scale() < old_scale)
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
                       train_mse=total/count, eta_epoch_seconds=(batches-batch-1)/rate,
                       best_val_mse=best if math.isfinite(best) else None)
        report(stage='검증', percent=100, batch=batches, batches=batches)
        val, psnr, sam = validate(model, validation, device)
        improved = val < best
        best = min(best, val)
        row = dict(epoch=epoch+1, train_mse=total/count, val_mse=val, val_psnr_band_mean_peak1=psnr,
                   val_sam_deg=sam, seconds_this_segment=time.time()-tick, skipped_amp_steps=skipped)
        history.append(row)
        print(json.dumps(row), flush=True)
        epoch += 1; next_batch = 0; total = 0.; count = 0
        if improved: checkpoint('best.pt')
        checkpoint(); last_save = time.time()
        if STOP or time.time() >= config['deadline']: break
    report(stage='완료' if epoch >= config['epochs'] else '세션 분량 완료 · 재개 가능', percent=100, completed_epochs=epoch,
           best_val_mse=best, next_batch=next_batch)
    if epoch >= config['epochs']:
        atomic_json(dict(epochs=epoch, best_val_mse=best, fingerprint=config['fingerprint']), folder/'complete.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--variant', choices=['baseline','mtf_aug'], required=True)
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
