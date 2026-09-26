from typing import Any
from .ecrformer_config import ECRformerConfig


class ECRformerLightConfig(ECRformerConfig):
    """Configuration for ECRformer-Light (lightweight variant)."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # only different from ECRformer in scale:
        self.net.cfg['features_start'] = 32
        self.net.cfg['num_blocks'] = [2, 2, 1, 1]
        self.net.cfg['num_refine'] = 2
        
        # equal to batch_size * accumulate_grad_batches = 16
        # avoid OOM for 24GB GPU while maintaining stable training
        self.train.train_bs = 8
        self.optim.accumulate_grad_batches = 2
