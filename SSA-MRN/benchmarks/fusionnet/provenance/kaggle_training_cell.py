# GF2 DLPan retrain, seed 42, 400 epochs; select by validation MSE only.
import random
seed=42; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
net=mod.FusionNet(4,None).cuda();opt=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=0)
run=pathlib.Path('/kaggle/working/fusionnet_gf2_retrain');run.mkdir(exist_ok=True)
paths={k:next(p for p in pathlib.Path(data).glob('*'+k+'_gf2.h5') if '._' not in p.name) for k in ('train','valid')}
def filehash(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
config={'model':'FusionNet DLPan official PyTorch','source_commit':sha,'seed':42,'sensor':'GF2','img_range':1023.,'epochs':400,'batch':32,'lr':3e-4,'optimizer':'Adam weight_decay=0','loss':'MSE','selection':'minimum validation MSE','scheduler':None,'deadline_seconds':28800,'files':{k:{'name':p.name,'sha256':filehash(p)} for k,p in paths.items()},'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'source_note':'DLPan config defaults, normalization and seed set for GF2'}
(run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
def readset(p):
 with h5py.File(p,'r') as h: return {k:np.asarray(h[k],dtype=np.float32)/1023. for k in ('gt','pan','lms')}
tr=readset(paths['train']);va=readset(paths['valid']);print('TRAINING SHAPES',[(k,v.shape) for k,v in tr.items()],flush=True)
best=float('inf');history=[];start=time.time()
def save(path,epoch):
 torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best_validation_mse':best,'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'config':config},path)
save(run/'initial.pt',0)
for epoch in range(1,401):
 net.train(); order=np.random.permutation(len(tr['gt']));total=0.;n=0
 for offset in range(0,len(order),32):
  ix=order[offset:offset+32]; b={k:torch.from_numpy(v[ix]).cuda() for k,v in tr.items()};pred=b['lms']+net(b['lms'],b['pan']);loss=((pred-b['gt'])**2).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=loss.item()*len(ix);n+=len(ix)
 net.eval();vtotal=0.;vn=0
 with torch.no_grad():
  for offset in range(0,len(va['gt']),32):
   b={k:torch.from_numpy(v[offset:offset+32]).cuda() for k,v in va.items()};v=((b['lms']+net(b['lms'],b['pan'])-b['gt'])**2).mean().item();vtotal+=v*len(b['gt']);vn+=len(b['gt'])
 score=vtotal/vn;improved=score<best;best=min(best,score);history.append({'epoch':epoch,'train_mse':total/n,'validation_mse':score,'seconds':time.time()-start})
 (run/'history.json').write_text(json.dumps(history,indent=2)+'\n')
 if improved: save(run/'best.pt',epoch)
 save(run/'latest.pt',epoch)
 print('GF2 EPOCH',history[-1],'best',best,flush=True)
 if time.time()-start>28800: print('DEADLINE: incomplete; use latest to resume',flush=True);break
(run/'status.json').write_text(json.dumps({'epochs_completed':epoch,'target_epochs':400,'complete':epoch==400,'best_validation_mse':best,'seconds':time.time()-start},indent=2)+'\n')
print('GF2 TRAINING RUN ENDED',flush=True)
