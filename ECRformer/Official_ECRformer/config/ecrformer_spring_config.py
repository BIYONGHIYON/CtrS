from typing import Any

from .ecrformer_config import ECRformerConfig


class EcrformerSpringConfig(ECRformerConfig):
    """Previous server settings; fixed subset is not season-balanced yet."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.dataset.root = r"E:\윤지\DataSet\ECRformer Data\SEN12MSCR_spring"
        self.train.train_bs = 2
        self.train.valid_bs = 1
        self.train.num_workers = 2
        self.train.max_train_samples = 6000
        self.train.max_epoch = 100
        self.optim.accumulate_grad_batches = 8
        self.optim.precision = "16-mixed"
