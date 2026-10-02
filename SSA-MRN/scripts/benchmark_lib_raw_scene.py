import sys,json,time,gc
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,'SSA-MRN/scripts')
import benchmark_lib_pipeline as bench
from ssamrn.data.lib_hsi import envi_metadata
from ssamrn.data.registration import warp_projective


class RawSceneLoader(bench.ThreadLoader):
    def prepare(self,indices):
        ids=[index for index,epoch in indices]
        scene=ids[0]//4
        assert len(indices)==4 and ids==list(range(scene*4,scene*4+4))
        data=self.dataset
        hdr=data.headers[scene];meta=envi_metadata(hdr)
        raw=np.fromfile(hdr.with_suffix('.dat'),dtype='<f4' if meta['byte order']==0 else '>f4',count=512*204*512,offset=meta['header offset'])
        if raw.size!=512*204*512 or not np.isfinite(raw).all():raise ValueError('Invalid raw cube')
        raw=raw.astype('float32',copy=False).reshape(512,204,512)
        np.clip(raw,0,1,out=raw)
        with Image.open(data.files[scene]) as im:rgb=np.array(im.convert('RGB'))
        rgb,valid=warp_projective(rgb,data.alignment['train/'+data.files[scene].stem]['matrix'])
        flips=[]
        for index,epoch in indices:
            generator=torch.Generator().manual_seed(data.crop_seed+epoch*len(data)+index)
            flips.append(tuple(bool(torch.rand((),generator=generator)<.5) for _ in range(2)))
        return {'raw_cube':torch.from_numpy(raw).pin_memory(),
                'rgb_full':torch.from_numpy(rgb.transpose(2,0,1).astype('float32')/255).pin_memory(),
                'valid_full':torch.from_numpy(valid).pin_memory(),'flips':flips,
                'scene':[data.files[scene].stem]*4,'source_cube_bytes':torch.tensor([raw.nbytes])}


def decode(host):
    raw=host['raw_cube'].to('cuda',non_blocking=True)
    full_gt=raw.permute(1,2,0).flip(-1)
    rgb=host['rgb_full'].to('cuda',non_blocking=True)
    valid=host['valid_full'].to('cuda',non_blocking=True)
    gt_patches,rgb_patches,masks=[],[],[]
    for tile,flips in enumerate(host['flips']):
        y,x=(tile//2)*256,(tile%2)*256
        gt_patch=full_gt[:,y:y+256,x:x+256]
        rgb_patch=rgb[:,y:y+256,x:x+256]
        mask=valid[y:y+256,x:x+256]
        axes=[axis+1 for axis,flag in enumerate(flips) if flag]
        if axes:
            gt_patch=gt_patch.flip(axes);rgb_patch=rgb_patch.flip(axes)
            mask=mask.flip([axis-1 for axis in axes])
        gt_patches.append(gt_patch);rgb_patches.append(rgb_patch);masks.append(mask)
    gt=torch.stack(gt_patches)
    return {'rgb':torch.stack(rgb_patches),'gt':gt,'lr_hsi':torch.nn.functional.avg_pool2d(gt,4),
            'valid_mask':torch.stack(masks),'scene':host['scene'],'source_cube_bytes':host['source_cube_bytes']}


class RawPrefetch:
    def __init__(self,loader,trainer):
        self.loader=loader
        self.stream=trainer.DevicePrefetch(loader,torch.device('cuda')).stream
    def __iter__(self):
        for host in self.loader:
            with torch.cuda.stream(self.stream):batch=decode(host)
            current=torch.cuda.current_stream();current.wait_stream(self.stream)
            for key in ('gt','rgb','lr_hsi','valid_mask'):batch[key].record_stream(current)
            yield batch


def main():
    import importlib.util
    spec=importlib.util.spec_from_file_location('raw_benchmark_trainer',bench.ROOT/'SSA-MRN/scripts/train_lib.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    config=json.loads((bench.ROOT/'SSA-MRN/configs/lib_rgb_hsi_grouped12_k4_23tap.json').read_text())
    torch.set_num_threads(config['cpu_threads']);torch.backends.cudnn.benchmark=True
    data=bench.dataset(config,16)
    loader=RawSceneLoader(data,4,2);loader.sampler.set_epoch(0)
    # Require exact CPU/GPU agreement across all pixels for two differently flipped scenes.
    batches=iter(loader.batches)
    for _ in range(2):
        indices=next(batches);candidate=decode(loader.prepare(indices))
        expected=torch.utils.data.default_collate([data[index] for index in indices])
        for key in ('rgb','gt','valid_mask'):
            torch.testing.assert_close(candidate[key].cpu(),expected[key],atol=0,rtol=0)
    del candidate,expected
    bench.close_loader(loader)
    torch.cuda.empty_cache()
    results=[]
    for mode in ('process:2','raw:2','raw:2','process:2'):
        loader=RawSceneLoader(bench.dataset(config,16),4,2) if mode=='raw:2' else bench.make_loader(config,mode,16)
        torch.manual_seed(config['seed'])
        model=trainer.build_rgb_hsi_model(config).cuda().train()
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],fused=True)
        scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats()
        records=[]
        for repetition in range(3):
            loader.sampler.set_epoch(0)
            batches=RawPrefetch(loader,trainer) if mode=='raw:2' else trainer.DevicePrefetch(loader,torch.device('cuda'))
            started=time.perf_counter();count=0
            for host in batches:
                rgb,lr,gt=host['rgb'],host['lr_hsi'],host['gt']
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):
                    prediction=model(rgb,lr)
                    mask=host['valid_mask'][:,None]
                    loss=((prediction.float()-gt).square()*mask).sum()/(mask.sum()*204)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
                scaler.step(optimizer);scaler.update();count+=gt.shape[0]
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            record=dict(mode=mode,repetition=repetition,seconds=seconds,patches_per_second=count/seconds,
                        last_loss=float(loss),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        current_reserved_mib=torch.cuda.memory_reserved()/2**20)
            records.append(record);print(json.dumps(record),flush=True)
        results.append(records)
        bench.close_loader(loader)
        del loader,model,optimizer,scaler,host,rgb,lr,gt,prediction,mask,loss
        gc.collect();torch.cuda.empty_cache()
    (bench.ROOT/'SSA-MRN/experiments/results/lib_raw_scene_candidate_benchmark.json').write_text(json.dumps({'exact_input_equality':True,'scenes':16,'batch_size':4,'dtype':'float32','order':'ABBA','results':results},indent=2))

if __name__=='__main__':main()
