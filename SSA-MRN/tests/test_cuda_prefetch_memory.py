"""Regression for stream-specific allocator pools growing once per epoch."""
import importlib.util
from pathlib import Path
import unittest
import torch

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PrefetchMemoryTests(unittest.TestCase):
    def test_new_iterators_reuse_stream_and_memory(self):
        path = Path(__file__).resolve().parents[1]/'scripts/train_lib.py'
        spec = importlib.util.spec_from_file_location('memory_trainer_test',path)
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        host = {'rgb':torch.zeros(1,3,128,128).pin_memory(),
                'gt':torch.ones(1,204,128,128).pin_memory(),'scene':['scene']}
        streams, reserved = [], []
        for _ in range(12):
            prefetch = trainer.DevicePrefetch([host,host],torch.device('cuda'))
            streams.append(prefetch.stream.cuda_stream)
            for batch in prefetch:
                self.assertEqual(batch['gt'].mean().item(),1)
            del batch
            torch.cuda.synchronize()
            reserved.append(torch.cuda.memory_reserved())
        self.assertEqual(len(set(streams)),1)
        self.assertLessEqual(max(reserved[2:])-min(reserved[2:]),32*2**20)

if __name__=='__main__': unittest.main()
