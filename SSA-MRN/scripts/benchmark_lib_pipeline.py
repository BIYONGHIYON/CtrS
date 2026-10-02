"""Compare lossless LIB loading paths without changing training checkpoints."""
import argparse
import copy
import gc
import importlib.util
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, BatchSampler, default_collate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'SSA-MRN/src'))
from ssamrn.data.lib_hsi import LIBHSI, ScenePatchSampler


def worker_init(_):
    torch.set_num_threads(1)


def pin(batch):
    if isinstance(batch, torch.Tensor):
        return batch.pin_memory()
    if isinstance(batch, dict):
        return {key: pin(value) for key, value in batch.items()}
    if isinstance(batch, list):
        return [pin(value) for value in batch]
    return batch


class ThreadLoader:
    """Candidate: isolated per-thread scene caches and bounded ordered prefetch."""
    def __init__(self, dataset, batch_size, workers=2):
        self.dataset = dataset
        self.sampler = ScenePatchSampler(dataset)
        self.batches = BatchSampler(self.sampler, batch_size, drop_last=False)
        self.workers = workers
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def __len__(self):
        return len(self.batches)

    def prepare(self, indices):
        if not hasattr(self.local, 'dataset'):
            dataset = copy.copy(self.dataset)
            dataset._cache_scene, dataset._cache_cube = None, None
            dataset._rgb_scene, dataset._stripe_scene = None, None
            dataset._stripe_cache = {}
            dataset._read_bytes_total = 0
            self.local.dataset = dataset
        return pin(default_collate([self.local.dataset[index] for index in indices]))

    def __iter__(self):
        iterator = iter(self.batches)
        pending = deque()
        for _ in range(self.workers):
            indices = next(iterator, None)
            if indices is not None:
                pending.append(self.pool.submit(self.prepare, indices))
        while pending:
            batch = pending.popleft().result()
            indices = next(iterator, None)
            if indices is not None:
                pending.append(self.pool.submit(self.prepare, indices))
            yield batch

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def dataset(config, limit):
    return LIBHSI(ROOT/config['data_root'], 'train', patch_size=config['patch_size'],
                  patches_per_scene=config['patches_per_scene'], limit=limit, compute_lr=False,
                  crop_seed=config['seed'], read_mode=config['read_mode'],
                  alignment_manifest=ROOT/config['alignment_manifest'],
                  train_layout=config['train_layout'], eval_layout=config['eval_layout'],
                  degradation=config['degradation'])


def make_loader(config, mode, limit):
    data = dataset(config, limit)
    kind, count = mode.split(':'); count = int(count)
    if kind == 'thread':
        return ThreadLoader(data, config['batch_size'], count)
    sampler = ScenePatchSampler(data)
    options = dict(batch_size=config['batch_size'], sampler=sampler, num_workers=count,
                   pin_memory=True)
    if count:
        options.update(persistent_workers=True, prefetch_factor=1, worker_init_fn=worker_init)
    return DataLoader(data, **options)


def close_loader(loader):
    if isinstance(loader, ThreadLoader):
        loader.close()
    elif loader.num_workers and loader._iterator is not None:
        loader._iterator._shutdown_workers()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'SSA-MRN/configs/lib_rgb_hsi_grouped12_k4_23tap.json')
    p.add_argument('--scenes', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--modes', nargs='+', default=['process:2', 'thread:2', 'process:1', 'process:0', 'thread:1'])
    p.add_argument('--train', action='store_true', help='Include actual CUDA optimizer steps in isolated temporary models')
    p.add_argument('--output', type=Path, default=ROOT/'SSA-MRN/experiments/results/lib_pipeline_benchmark.json')
    args = p.parse_args()
    if args.scenes < 1 or args.repeats < 2:
        p.error('Use positive scenes and at least two repeats (first is warmup)')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    torch.set_num_threads(config['cpu_threads'])
    if not torch.cuda.is_available():
        p.error('CUDA needed for pinned-memory comparison')
    spec = importlib.util.spec_from_file_location('lib_pipeline_trainer', ROOT/'SSA-MRN/scripts/train_lib.py')
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    results = []
    expected_signature = None
    for mode in args.modes:
        loader = make_loader(config, mode, args.scenes)
        if args.train:
            torch.manual_seed(config['seed'])
            model = trainer.build_rgb_hsi_model(config).cuda().train()
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], fused=True)
            scaler = torch.amp.GradScaler('cuda')
            torch.backends.cudnn.benchmark = config['cudnn_benchmark']
            torch.cuda.reset_peak_memory_stats()
        measurements = []
        try:
            for repetition in range(args.repeats):
                loader.sampler.set_epoch(0)  # Exactly the same scenes, crops, flips and order.
                started = time.perf_counter()
                signature, samples, byte_count, waits, gpu_events = [], 0, 0, 0., []
                batches = trainer.DevicePrefetch(loader, torch.device('cuda')) if args.train else loader
                last = time.perf_counter()
                for host in batches:
                    waits += time.perf_counter()-last
                    samples += host['gt'].shape[0]
                    byte_count += int(host['source_cube_bytes'].sum())
                    if args.train:
                        first, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        first.record()
                        rgb, lr, gt = trainer.move_batch(host, torch.device('cuda'))
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda'):
                            prediction = model(rgb, lr)
                            mask = host['valid_mask'][:, None]
                            loss = ((prediction.float()-gt).square()*mask).sum()/(mask.sum()*204)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
                        scaler.step(optimizer); scaler.update()
                        end.record(); gpu_events.append((first, end))
                    else:
                        # Check ordering and deterministic crop/flip content outside GPU work.
                        signature.extend(zip(host['scene'], host['gt'][:, 0, 0, 0].tolist(),
                                             host['rgb'][:, 0, 0, 0].tolist()))
                    last = time.perf_counter()
                    del host
                if args.train:
                    torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                if not args.train:
                    if expected_signature is None:
                        expected_signature = signature
                    if signature != expected_signature:
                        raise RuntimeError('Loader changed scene order or input values')
                record = dict(repetition=repetition, seconds=seconds, patches_per_second=samples/seconds,
                              source_cube_requested_gib=byte_count/2**30, batch_wait_seconds=waits)
                if args.train:
                    record.update(gpu_stream_seconds=sum(a.elapsed_time(b) for a,b in gpu_events)/1000,
                                  loss=float(loss), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                  current_reserved_mib=torch.cuda.memory_reserved()/2**20)
                measurements.append(record)
                print(json.dumps(dict(mode=mode, **record)), flush=True)
        finally:
            close_loader(loader)
        results.append(dict(mode=mode, measurements=measurements))
        del loader
        if args.train:
            del model, optimizer, scaler, rgb, lr, gt, prediction, loss, mask
        gc.collect(); torch.cuda.empty_cache()
    report = dict(config=str(args.config), train=args.train, scenes=args.scenes,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), results=results,
                  note='First repetition includes startup. OS cache is not flushed. Same float32 inputs and sampler epoch. Loader-only throughput excludes model work; subset is not full-epoch speed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
