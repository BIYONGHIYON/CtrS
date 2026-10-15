import sys,importlib,json
from pathlib import Path
S=Path('C:/CtrS-ssai8/SSA-MRN');R=Path('C:/CtrS-ssai8/runtime/qb_s42_100_verified');sys.path.insert(0,str(S))
from common import ssai_suite as s
import torch
d=importlib.import_module('experiments.ssai.8_spatial_mean_normalization.diagnostics')
s.configure(42,'cpu');m=s.model('A0');state=m.state_dict();plan=json.loads((R/'plan.json').read_text());h=d.state_hash(state);assert h==plan['initial_common_sha256']
s.save(R/'final_delivery/provenance/initial_state.pt',{'model':state,'seed':42,'torch':torch.__version__,'sha256_tensor_state':h,'reconstructed_with':'actual training Python/Torch/source; no optimizer update'})
print(h,flush=True)
