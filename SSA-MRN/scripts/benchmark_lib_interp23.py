import sys,time,json
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,'SSA-MRN/src')
from ssamrn.models.interp23 import interp23tap

def polyphase(image,ratio=4):
    half=[.5,.305334091185,0,-.072698593239,0,.021809577942,0,-.005192756653,0,.000807762146,0,-.000060081482]
    with torch.autocast(image.device.type,enabled=False):
        image=image.float();kernel=(image.new_tensor(half[:0:-1]+half)*2)[::2]
        c=image.shape[1]
        for stage in range(ratio.bit_length()-1):
            offset=1 if stage==0 else 0
            left=6 if offset==1 else 5
            n,c,h,w=image.shape
            rows=torch.arange(-left,h+11-left,device=image.device)%h
            interpolated=F.conv2d(image.index_select(2,rows),kernel.view(1,1,12,1).expand(c,1,12,1).contiguous(),groups=c)
            up=image.new_empty(n,c,2*h,w)
            up[:,:,offset::2,:]=image;up[:,:,1-offset::2,:]=interpolated
            cols=torch.arange(-left,w+11-left,device=image.device)%w
            interpolated=F.conv2d(up.index_select(3,cols),kernel.view(1,1,1,12).expand(c,1,1,12).contiguous(),groups=c)
            out=image.new_empty(n,c,2*h,2*w)
            out[:,:,:,offset::2]=up;out[:,:,:,1-offset::2]=interpolated
            image=out
    return image

def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=True
    for ratio in (2,4):
        a=torch.rand(2,3,7,9,requires_grad=True);b=a.detach().clone().requires_grad_()
        x,y=interp23tap(a,ratio),polyphase(b,ratio)
        torch.testing.assert_close(y,x,atol=1e-6,rtol=2e-6)
        x.square().sum().backward();y.square().sum().backward()
        torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=2e-5)
    results=[]
    for c,size,ratio in ((204,64,4),(12,64,4),(12,128,2)):
        sample=torch.rand(4,c,size,size,device='cuda')
        torch.testing.assert_close(polyphase(sample,ratio),interp23tap(sample,ratio),atol=3e-6,rtol=2e-5)
        times={}
        for name,method in [('original',interp23tap),('polyphase',polyphase),('polyphase',polyphase),('original',interp23tap)]:
            for _ in range(4):method(sample,ratio)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(20):method(sample,ratio)
            torch.cuda.synchronize();seconds=(time.perf_counter()-start)/20
            times.setdefault(name,[]).append(seconds)
        record=dict(channels=c,lr_size=size,ratio=ratio,batch_size=4,seconds_per_call=times)
        results.append(record);print(json.dumps(record),flush=True)
    Path('SSA-MRN/experiments/results/lib_interp23_candidate_benchmark.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
