import sys, unittest
import tempfile, importlib.util
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, shift
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ssamrn.data.registration import estimate, warp_rgb

class RegistrationTests(unittest.TestCase):
    def test_invalid_border_excluded_from_metrics(self):
        import torch
        script=Path(__file__).resolve().parents[1]/'scripts/train_lib.py'
        spec=importlib.util.spec_from_file_location('registration_trainer',script)
        trainer=importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
        class Model(torch.nn.Module):
            def forward(self,rgb,lr):
                out=torch.ones(1,204,8,8); out[:,:,0]=100
                return out
        mask=torch.ones(1,8,8,dtype=torch.bool); mask[:,0]=False
        batch={'rgb':torch.ones(1,3,8,8),'lr_hsi':torch.ones(1,204,2,2),
               'gt':torch.ones(1,204,8,8),'scene':['test'],'valid_mask':mask}
        with tempfile.TemporaryDirectory() as d:
            result=trainer.evaluate(Model(),[batch],torch.device('cpu'),False,Path(d))
        self.assertEqual(result['scene_mean']['model_mse'],0)

    def test_known_translation_and_no_spectral_resampling(self):
        rng=np.random.default_rng(42)
        reference=gaussian_filter(rng.random((256,256)),1)
        reference=np.repeat(reference[:,:,None],3,axis=2)
        moving=shift(reference,(4,-3,0),order=0,mode='constant')
        result=estimate(reference,moving)
        self.assertTrue(result['accepted'])
        self.assertEqual((result['dy'],result['dx']),(-4,3))
        aligned,valid=warp_rgb(moving,-4,3)
        np.testing.assert_array_equal(aligned[valid],reference[valid])
        self.assertFalse(valid[-1].any())

    def test_identity_and_unrelated_pair_not_warped(self):
        rng=np.random.default_rng(1)
        image=rng.random((256,256,3)).astype('float32')
        self.assertFalse(estimate(image,image)['accepted'])
        result=estimate(image,rng.random(image.shape))
        self.assertFalse(result['accepted'])
        self.assertEqual((result['dy'],result['dx']),(0,0))
        aligned,valid=warp_rgb(image,0,0)
        np.testing.assert_array_equal(aligned,image)
        self.assertTrue(valid.all())

if __name__=='__main__': unittest.main()
