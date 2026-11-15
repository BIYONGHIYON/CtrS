import torch
class EMA:
    """FP32 shadows, initialized exactly from online weights; update after each successful step.
    Floating buffers are averaged, integer buffers copied. Shadows never feed back into Adam.
    """
    def __init__(self,model,betas):
        assert len(set(betas))==len(betas) and all(0<b<1 for b in betas)
        self.betas=list(betas);self.updates=0
        self.shadows={str(b):{k:v.detach().clone() for k,v in model.state_dict().items()} for b in betas}
    @torch.no_grad()
    def update(self,model):
        for key,state in self.shadows.items():
            b=float(key)
            for name,v in model.state_dict().items():
                if v.is_floating_point():state[name].mul_(b).add_(v.detach(),alpha=1-b)
                else:state[name].copy_(v)
        self.updates+=1
    def state_dict(self):
        return {'betas':self.betas,'updates':self.updates,'shadows':{b:{k:v.detach().cpu().clone() for k,v in s.items()} for b,s in self.shadows.items()}}
    def load_state_dict(self,value):
        if value['betas']!=self.betas:raise ValueError('EMA beta mismatch')
        if set(value['shadows'])!=set(self.shadows):raise ValueError('EMA shadow mismatch')
        for b,s in self.shadows.items():
            for k,v in s.items():v.copy_(value['shadows'][b][k].to(v.device))
        self.updates=value['updates']
