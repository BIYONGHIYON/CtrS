import sys,time,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from run import *
a=sys.argv[1:];root=Path('C:/CtrS-p150/runtime');plan=json.loads((root/'QB/plan.json').read_text())
def main():
    if a:
        v=a[0];seed();m=StructureNet(v);m.configure(plan['calibration']);m=m.cuda().train();o=torch.optim.Adam(m.parameters(),lr=1e-4);scaler=torch.amp.GradScaler('cuda');ds=Cached(Path(plan['cache_root'])/'train');dl=loader(ds,1,Remaining(len(ds),0,0,32));t=time.monotonic();times=[]
        for i,b in enumerate(dl):
            b=[x.cuda(non_blocking=True) for x in b];o.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):p=m(*b[:3])
            scaler.scale((p.float()-b[3]).square().mean()).backward();scaler.step(o);scaler.update()
            if i==9:torch.cuda.synchronize();start=time.monotonic()
            if i==69:break
        torch.cuda.synchronize();write(root/('bench_'+v+'.json'),dict(variant=v,step_seconds=(time.monotonic()-start)/60,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,seconds=time.monotonic()-t));print(v,flush=True)
    else:
        for group in [('R0','W2'),('W3','F01'),('F11',)]:
            ps=[subprocess.Popen([sys.executable,__file__,v]) for v in group]
            for p in ps:
                if p.wait():raise RuntimeError('Benchmark failed')
        records={v:json.loads((root/('bench_'+v+'.json')).read_text()) for v in VARIANTS}
        # Pair measurements include competition; the single remaining job uses its solo measurement.
        slots=[0.,0.]
        for v in VARIANTS:
            seconds=records[v]['step_seconds']*536*150*1.18
            k=min(range(2),key=lambda i:slots[i]);slots[k]+=seconds
        write(root/'benchmark.json',dict(runs=records,projected_hours=max(slots)/3600,includes_validation_checkpoint_factor=1.18));print(json.dumps(records),max(slots)/3600,flush=True)

if __name__=='__main__':main()
