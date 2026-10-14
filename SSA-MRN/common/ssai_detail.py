"""Two SSAI-only residual detail hypotheses with unchanged original initialization."""
import torch
from torch import nn
from torch.nn import functional as F
from common.model import RestoredPansharpeningNet

def highpass(x):
    return x-F.avg_pool2d(F.pad(x,(1,1,1,1),mode="reflect"),3,stride=1)

class DetailSSA(nn.Module):
    def __init__(self,source,area,channel_relation):
        super().__init__()
        self.fixed=source.fixed
        self.reference_area=area
        self.channel_relation=channel_relation
        for name,child in source.named_children():
            self.add_module(name,child)
        self.detail_out=nn.Conv2d(4,4,1,bias=False)
        nn.init.zeros_(self.detail_out.weight)

    def forward(self,pan,ms_pro):
        p=self.conv6t6(self.relu(self.conv1t6(pan)))
        q=self.conv7t6_3(ms_pro)
        b,c,h,w=q.shape
        z=p.reshape(b,c,-1)*q.transpose(-1,-2).reshape(b,c,-1)
        z=z.reshape(b,c,h,w).transpose(-1,-2).reshape(b,c,-1)
        original=(p.reshape(b,c,-1)*z.softmax(dim=-1)).reshape(b,c,h,w)
        detail=highpass(p)
        if self.channel_relation:
            query=F.normalize(q.flatten(2),dim=-1,eps=1e-6)
            key=F.normalize(detail.flatten(2),dim=-1,eps=1e-6)
            relation=torch.softmax(torch.bmm(query,key.transpose(1,2))/0.25,dim=-1)
            detail=torch.bmm(relation,detail.flatten(2)).reshape(b,c,h,w)
        gate=2*q.sigmoid()
        correction=highpass(self.detail_out(gate*detail))/self.reference_area
        return original+correction

class DetailNet(RestoredPansharpeningNet):
    def __init__(self,channels=4,ssai_dimension=4,channel_relation=False):
        if channels!=4 or ssai_dimension!=4:raise ValueError("Initial study fixes QB C4 and K4")
        super().__init__(channels,ssai_dimension)
        # Allocate added zero weights without advancing the original global RNG.
        with torch.random.fork_rng(devices=[]):
            for name,area in zip(("SSA_blocks","SSA_blocks1","SSA_blocks2"),(4096,1024,256)):
                blocks=getattr(self,name)
                for band,source in enumerate(blocks):
                    blocks[band]=DetailSSA(source,area,channel_relation)

    def forward(self,pan,lms,ms):
        if self.training and tuple(lms.shape[-2:])!=(64,64):raise ValueError("Training reference must remain64x64")
        return super().forward(pan,lms,ms)
