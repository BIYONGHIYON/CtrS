"""Same N0 parameters, different sampling geometry in two active backbone convolutions."""
from normalization33 import Model as ArchivedModel

TARGETS=('res2.conv21','res4.conv21')

def architecture_spec(arm):
    if arm not in ('baseline','dilated'):raise ValueError('Expected baseline or dilated')
    return {'arm':arm,'model':'archived33 N0','targets':['core.'+p for p in TARGETS],
            'aliases':['core.backbone.1.conv21','core.backbone.3.conv21'],
            'kernel':[3,3],'stride':[1,1],'dilation':[2,2] if arm=='dilated' else [1,1],
            'padding':[2,2] if arm=='dilated' else [1,1],'padding_mode':'zeros',
            'weights_unchanged_shape':True,'parameters':426654,'initial_outputs_expected_equal':False}

class Model(ArchivedModel):
    def __init__(self,arm='baseline'):
        if arm not in ('baseline','dilated'):raise ValueError('Expected baseline or dilated')
        super().__init__('N0')
        self.experiment_arm=arm
        for path,index in zip(TARGETS,(1,3)):
            conv=self.core.get_submodule(path)
            if conv is not self.core.backbone[index].conv21:raise RuntimeError('Backbone alias changed')
            if conv.kernel_size!=(3,3) or conv.stride!=(1,1) or conv.dilation!=(1,1) or conv.padding!=(1,1) or conv.padding_mode!='zeros':
                raise RuntimeError('Unexpected archived convolution geometry')
            if arm=='dilated':
                # Metadata only: preserve the original Parameter objects and RNG stream.
                conv.dilation=(2,2);conv.padding=(2,2)
        if sum(p.numel() for p in self.parameters())!=426654:raise RuntimeError('Unexpected parameter count')

def common_state(model):return model.state_dict()
def load_common(model,state):model.load_state_dict(state,strict=True)
