import sys,json,time,gc,importlib.util
from pathlib import Path
import torch
import benchmark_lib_raw_scene as raw
from benchmark_lib_interp23 import polyphase
import ssamrn.models.interp23 as interpolation
import ssamrn.models.rgb_grouped as grouped

def main():
    config=json.loads(Path('SSA-MRN/configs/lib_rgb_hsi_grouped12_k4_23tap.json').read_text())
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=True
    spec=importlib.util.spec_from_file_location('combined_benchmark_trainer',raw.bench.ROOT/'SSA-MRN/scripts/train_lib.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    original=interpolation.interp23tap
    results=[]
    for mode in ('raw_original23','raw_polyphase23','raw_polyphase23','raw_original23'):
        method=polyphase if mode=='raw_polyphase23' else original
        interpolation.interp23tap=method;grouped.interp23tap=method
        loader=raw.RawSceneLoader(raw.bench.dataset(config,16),4,2)
        torch.manual_seed(config['seed']);model=trainer.build_rgb_hsi_model(config).cuda().train()
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],fused=True)
        scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats()
        records=[]
        for repetition in range(3):
            loader.sampler.set_epoch(0);started=time.perf_counter()
            for host in raw.RawPrefetch(loader,trainer):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):
                    prediction=model(host['rgb'],host['lr_hsi'])
                    mask=host['valid_mask'][:,None]
                    loss=((prediction.float()-host['gt']).square()*mask).sum()/(mask.sum()*204)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
                scaler.step(optimizer);scaler.update()
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            record=dict(mode=mode,repetition=repetition,seconds=seconds,patches_per_second=64/seconds,
                        last_loss=float(loss),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        current_reserved_mib=torch.cuda.memory_reserved()/2**20)
            records.append(record);print(json.dumps(record),flush=True)
        results.append(records);raw.bench.close_loader(loader)
        del loader,model,optimizer,scaler,host,prediction,mask,loss
        gc.collect();torch.cuda.empty_cache()
    Path('SSA-MRN/experiments/results/lib_raw_polyphase_combined_benchmark.json').write_text(json.dumps({'scenes':16,'batch_size':4,'order':'ABBA','results':results},indent=2))

if __name__=='__main__':main()
