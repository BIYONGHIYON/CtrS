
import json
import os
import sys
from pathlib import Path
code = Path(sys.argv[1]).resolve(); dest = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(code / "SSA-MRN/src")); sys.path.insert(0, str(code))
import h5py
import torch
from alpha_models import build, state_hash, common_hash
from alpha_support import sha, write
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
models = {}
for arm in ("B1", "B1_alpha_bounded"):
    torch.manual_seed(42); models[arm] = build(arm).eval()
left, right = models.values()
if common_hash(left) != common_hash(right): raise RuntimeError("Core/encoder initialization differs")
for name, value in left.state_dict().items():
    if name.startswith(("core.", "error_encoder.")):
        if not torch.equal(value, right.state_dict()[name]): raise RuntimeError(name)
with h5py.File(sys.argv[3], "r") as f:
    inputs = [torch.from_numpy(f[k][:2]).float() / 2047. for k in ("pan", "lms", "ms")]
with torch.inference_mode():
    outputs = {arm: model(*inputs) for arm, model in models.items()}
torch.testing.assert_close(outputs["B1"], outputs["B1_alpha_bounded"], atol=1e-6, rtol=1e-5)
alphas = {arm: float(model.effective_alpha()) for arm, model in models.items()}
if any(abs(a - 0.1) > 1e-7 for a in alphas.values()): raise RuntimeError("Initial effective alpha mismatch")
states = {arm: model.state_dict() for arm, model in models.items()}
dest.mkdir(parents=True, exist_ok=True)
torch.save(dict(states=states, inputs=inputs), dest / "initial_states.pt")
proof = dict(seed=42, core_sha256=state_hash(left.core.state_dict()),
             encoder_sha256=state_hash(left.error_encoder.state_dict()), common_sha256=common_hash(left),
             alpha=alphas, output_max_abs=float((outputs["B1"] - outputs["B1_alpha_bounded"]).abs().max()),
             output_atol=1e-6, output_rtol=1e-5, alpha_atol=1e-7,
             input="first two validation patches; normalized peak2047; FP32 CPU",
             initial_states_sha256=sha(dest / "initial_states.pt"))
write(dest / "initialization_pair.json", proof)
print("CTRS_ALPHA_INIT=" + json.dumps(proof), flush=True)
