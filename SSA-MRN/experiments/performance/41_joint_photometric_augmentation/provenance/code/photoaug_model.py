"""Identical archived33 N0 networks in both arms; augmentation belongs to training data."""
from normalization33 import Model as ArchivedModel

class Model(ArchivedModel):
    def __init__(self,arm='baseline'):
        if arm not in ('baseline','aug'):raise ValueError('Expected baseline or aug')
        super().__init__('N0')
        self.experiment_arm=arm

def common_state(model):return model.state_dict()
def load_common(model,state):model.load_state_dict(state,strict=True)
