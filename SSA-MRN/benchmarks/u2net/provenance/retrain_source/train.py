"""One sensor worker. Checkpoints only at completed optimizer updates; no partial gradients saved."""
import argparse,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from model import build_network
from data import arrays
from utils import write,save,digest,stable,tensors_hash,batch_order,rng_state,restore_rng

def batch(a,idx,device):
    return {k:torch.from_numpy(np.array(v[idx],dtype=np.float32,copy=True)).to(device) for k,v in a.items()}

def validation(net,a,device,micro):
    total=0.;l1=0.;count=0;angle_sum=0.;angle_count=0;psnr_sum=0.;patch_count=0
    net.eval()
    with torch.no_grad():
        for start in range(0,len(a['gt']),micro):
            b=batch(a,slice(start,start+micro),device);pred=net(b['ms'],b['pan']);d=pred-b['gt']
            if not torch.isfinite(d).all():raise FloatingPointError('Nonfinite validation')
            total+=float(d.square().sum());l1+=float(d.abs().sum());count+=d.numel()
            ref=b['gt'].double();out=pred.double();norm=ref.norm(dim=1)*out.norm(dim=1);valid=norm>0
            angle=torch.rad2deg(torch.acos(((ref*out).sum(1)[valid]/norm[valid]).clamp(-1,1)))
            angle_sum+=float(angle.sum());angle_count+=angle.numel()
            band_mse=(out-ref).square().mean((-2,-1)).clamp_min(1e-30)
            psnr_sum+=float((-10*band_mse.log10()).mean(1).sum());patch_count+=len(ref)
    return dict(mse=total/count,l1=l1/count,psnr_normalized_mse=float(-10*np.log10(max(total/count,1e-30))),psnr_sensor_peak=psnr_sum/patch_count,sam_degrees=angle_sum/angle_count if angle_count else None)

def train(folder,cache,sensor,cfg,proof,device,deadline=float('inf'),stop_file=None,max_updates=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    random.seed(cfg['seed']+200000);np.random.seed(cfg['seed']+200000);torch.manual_seed(cfg['seed']+200000)
    net=build_network(4,cfg['seed']).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=100,gamma=.5)
    fingerprint=stable(dict(config=cfg,data=proof['data'],source=proof['source_sha256'],sensor=sensor))
    a=arrays(cache,sensor,'train');v=arrays(cache,sensor,'val')
    if cfg.get('smoke_mode'):
        a={k:x[:cfg['smoke']['train_samples']] for k,x in a.items()};v={k:x[:cfg['smoke']['validation_samples']] for k,x in v.items()}
    effective=cfg['effective_batch'];micro=cfg['micro_batch'];total_batches=len(a['gt'])//effective
    if total_batches==0:raise ValueError('Not enough samples for one effective batch')
    state=dict(epoch_completed=0,batch_cursor=0,updates=0,train_abs_sum=0.,train_sq_sum=0.,train_elements=0,history=[],best_mse=float('inf'),best_epoch=0,train_seconds=0.,resumes=0)
    latest=folder/'latest.pt';started=time.monotonic();session_updates=0
    def snapshot():
        return dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),state=state,rng=rng_state(),fingerprint=fingerprint,config=cfg,seed=cfg['seed'],sensor=sensor)
    def checkpoint():save(latest,snapshot())
    def status(phase,**extra):
        write(folder/'status.json',dict(sensor=sensor,phase=phase,epoch=state['epoch_completed']+(1 if phase in ('training','validation') else 0),epochs=cfg['epochs'],batch=state['batch_cursor'],batches=total_batches,updates=state['updates'],best_val_mse=state['best_mse'],lr=opt.param_groups[0]['lr'],gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3 if device.startswith('cuda') else 0,**extra))
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['fingerprint']!=fingerprint:raise ValueError('Resume config/data/source fingerprint mismatch')
        net.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);state=ck['state'];restore_rng(ck['rng']);state['resumes']+=1
    else:
        save(folder/'initial.pt',dict(model=net.state_dict(),seed=cfg['seed'],sensor=sensor,config=cfg,tensor_sha256=tensors_hash(net.state_dict())))
        status('initial validation');val=validation(net,v,device,micro);state.update(best_mse=val['mse'],best_epoch=0)
        save(folder/'best.pt',dict(model=net.state_dict(),seed=cfg['seed'],sensor=sensor,epoch=0,val=val,fingerprint=fingerprint,config=cfg));checkpoint()
    initial_hash=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)['tensor_sha256']
    write(folder/'initialization.json',dict(seed=cfg['seed'],model_tensor_sha256=initial_hash,sampling='independent seed+100000+epoch CPU generator'))
    train_time_start=time.monotonic()
    while state['epoch_completed']<cfg['epochs']:
        epoch=state['epoch_completed']+1;order,order_sha=batch_order(len(a['gt']),cfg['seed'],epoch,effective)
        while state['batch_cursor']<total_batches:
            if time.time()>=deadline or (stop_file and Path(stop_file).exists()) or (max_updates is not None and session_updates>=max_updates):
                state['train_seconds']+=time.monotonic()-train_time_start;checkpoint();status('paused');return 'paused'
            net.train();pos=state['batch_cursor']*effective;ids=order[pos:pos+effective];opt.zero_grad(set_to_none=True)
            abs_sum=sq_sum=0.;elements=0;denom=effective*np.prod(a['gt'].shape[1:])
            for offset in range(0,effective,micro):
                b=batch(a,ids[offset:offset+micro],device);d=net(b['ms'],b['pan'])-b['gt']
                if not torch.isfinite(d).all():raise FloatingPointError('Nonfinite training')
                loss=d.abs().sum()/denom;loss.backward();abs_sum+=float(d.detach().abs().sum());sq_sum+=float(d.detach().square().sum());elements+=d.numel()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()):raise FloatingPointError('Nonfinite gradient')
            opt.step();state['batch_cursor']+=1;state['updates']+=1;session_updates+=1
            state['train_abs_sum']+=abs_sum;state['train_sq_sum']+=sq_sum;state['train_elements']+=elements
            if state['updates']%cfg['checkpoint_every_updates']==0:checkpoint()
            elapsed=state['train_seconds']+time.monotonic()-train_time_start
            remaining=(cfg['epochs']*total_batches-state['updates'])*elapsed/max(1,state['updates'])/3600
            status('training',train_l1=state['train_abs_sum']/state['train_elements'],order_sha256=order_sha,remaining_hours=remaining)
        status('validation');val=validation(net,v,device,micro)
        row=dict(epoch=epoch,train_l1=state['train_abs_sum']/state['train_elements'],train_mse=state['train_sq_sum']/state['train_elements'],validation=val,lr=opt.param_groups[0]['lr'],order_sha256=order_sha,used_samples=len(order),dropped_samples=len(a['gt'])-len(order),updates=state['updates'])
        if val['mse']<state['best_mse']:
            state.update(best_mse=val['mse'],best_epoch=epoch)
            save(folder/'best.pt',dict(model=net.state_dict(),seed=cfg['seed'],sensor=sensor,epoch=epoch,val=val,fingerprint=fingerprint,config=cfg))
        state['history'].append(row);sched.step();state.update(epoch_completed=epoch,batch_cursor=0,train_abs_sum=0.,train_sq_sum=0.,train_elements=0)
        state['train_seconds']+=time.monotonic()-train_time_start;train_time_start=time.monotonic()
        checkpoint();write(folder/'history.json',state['history']);status('epoch complete',val_mse=val['mse'])
    write(folder/'training_complete.json',dict(fingerprint=fingerprint,epochs=cfg['epochs'],updates=state['updates'],best_epoch=state['best_epoch'],best_sha256=digest(folder/'best.pt'),latest_sha256=digest(latest),initial_sha256=digest(folder/'initial.pt'),train_seconds=state['train_seconds'],resumes=state['resumes'],gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3 if device.startswith('cuda') else None,smoke_excluded=bool(cfg.get('smoke_mode'))))
    status('training complete');return 'complete'

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--max-updates',type=int);a=p.parse_args();job=json.loads(a.job.read_text(encoding='utf8'))
    try:train(job['folder'],job['cache'],job['sensor'],job['config'],job['proof'],job['device'],job['deadline'],job['stop_file'],a.max_updates)
    except Exception:
        write(Path(job['folder'])/'error.json',dict(error=traceback.format_exc()));raise
if __name__=='__main__':main()
