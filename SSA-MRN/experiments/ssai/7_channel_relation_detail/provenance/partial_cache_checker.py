import sys,json,gc
from pathlib import Path
import torch
sys.path.insert(0,'C:/CtrS-ssai-detail/SSA-MRN')
from common import ssai_suite as s
s.configure(42,'cuda');ds=s.Cached('C:/CtrS-ssai4/runtime/qb_s42_100/cache/train')
g1=torch.Generator().manual_seed(42);g2=torch.Generator().manual_seed(42)
a=s.GPUCachedLoader(ds,32,g1,'cuda');b=s.PartialGPUCachedLoader(ds,32,g2,'cuda')
for epoch in range(2):
 seen=0
 ia,ib=iter(a),iter(b)
 for x,y in zip(ia,ib):
  assert all(torch.equal(u,v)for u,v in zip(x,y));seen+=len(x[0])
 assert not list(ia)and not list(ib)
 assert seen==17139 and torch.equal(g1.get_state(),g2.get_state())
del a,b;gc.collect();torch.cuda.empty_cache()
# Also prove equal numeric optimizer trajectory with the cache placement changed.
steps={}
for mode in ('gpu','gpu_partial'):
 s.configure(42,'cuda');net=s.model('A0').cuda();opt=torch.optim.Adam(net.parameters(),lr=1e-4)
 loader=s.make_loader(ds,mode,32,torch.Generator().manual_seed(42),'cuda')
 for i,batch in enumerate(loader):
  pan,lms,ms,gt=batch;opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(net(pan,lms,ms),gt);loss.backward();opt.step()
  if i==2:break
 steps[mode]={k:v.detach().cpu().clone()for k,v in net.state_dict().items()}
 del net,opt,loader,batch;gc.collect();torch.cuda.empty_cache()
assert all(torch.equal(v,steps['gpu_partial'][k])for k,v in steps['gpu'].items())
r={'two_epochs_samples_per_epoch':17139,'all_fp32_batches_exact_equal':True,'epoch_boundary_loader_rng_exact_equal':True,'three_adam_updates_state_exact_equal':True,'cache_fraction':.85,'conditions':'same micro32/FP32/data/order; only cache placement changed'}
Path('C:/CtrS-ssai-detail/partial_cache_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
