"""MS 조건부 공간 세부정보 보정."""
from common.ssai_detail import DetailNet

class MSConditionedDetailNet(DetailNet):
    def __init__(self,channels=4,ssai_dimension=4):
        super().__init__(channels,ssai_dimension,channel_relation=False)
