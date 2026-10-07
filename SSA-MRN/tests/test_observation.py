"""Check MTF numerics, flip geometry, guarded profiles and CUDA gradients."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from scipy.ndimage import correlate
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ssamrn.observation import SensorObservation,load_profile,module_hash


class ObservationTests(unittest.TestCase):
    def test_cpu_matches_independent_scipy_correlation(self):
        op=SensorObservation("QB").double()
        x=np.random.default_rng(42).normal(size=(1,4,64,64))
        expected=np.stack([correlate(x[0,c],op.kernel[c,0].numpy(),mode="nearest")[2::4,2::4] for c in range(4)])[None]
        actual=op(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

    def test_flipped_pairs_need_changed_phase(self):
        op=SensorObservation("QB").double()
        x=torch.randn(1,4,64,64,dtype=torch.float64)
        ms=op(x)
        for axis,phase in [(-2,[1,2]),(-1,[2,1])]:
            x_flip=x.flip(axis);ms_flip=ms.flip(axis)
            loss=op.loss(x_flip,ms_flip,5,[phase])
            self.assertLess(loss.item(),1e-24)
            self.assertGreater(op.loss(x_flip,ms_flip,5,[[2,2]]).item(),1e-8)

    def test_source_and_profile_hashes_guard_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);data=folder/"train.h5";data.write_bytes(b"fixture")
            stat=data.stat();p=folder/"profile.json"
            config=dict(status="validated",sensor="QB",operator_sha256=module_hash(),train_path=str(data),train_stat=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),phase_policy="fixed_per_training_sample",phases=[[2,2]],sample_count=1,coverage=1,sampling="decimate",ratio=4)
            p.write_text(json.dumps(config));digest=hashlib.sha256(p.read_bytes()).hexdigest()
            load_profile(p,digest,"QB",data)
            with self.assertRaises(ValueError):load_profile(p,"invalid","QB",data)
            config["operator_sha256"]="old code";p.write_text(json.dumps(config))
            with self.assertRaises(ValueError):load_profile(p,sensor="QB",train_path=data)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
    def test_deterministic_cuda_loss_backward(self):
        torch.use_deterministic_algorithms(True)
        op=SensorObservation("QB").float().cuda()
        prediction=torch.randn(3,4,64,64,device="cuda",requires_grad=True)
        target=torch.zeros(3,4,16,16,device="cuda")
        loss=op.loss(prediction,target,5,[[2,2],[1,2],[2,1]])
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(),0.)

if __name__=="__main__":unittest.main()
