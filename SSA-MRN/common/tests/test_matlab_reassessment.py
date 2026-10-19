"""Regression checks for retry and stale-result bugs; no training or CUDA needed."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
worker=load('matlab_reassessment_worker');controller=load('matlab_reassessment_controller')
class ResumeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  for d in ('inputs','queue','records','done'):(self.root/d).mkdir()
 def tearDown(self):self.tmp.cleanup()
 def test_error_record_is_retried(self):
  p=self.root/'records/a.json';p.write_text(json.dumps(dict(version=worker.VERSION,status='error')))
  self.assertFalse(worker.reusable(p,self.root))
 def test_corrupt_export_is_not_reused(self):
  q=self.root/'queue/a.mat';i=self.root/'inputs/data.mat';q.write_bytes(b'prediction');i.write_bytes(b'observation')
  record=dict(version=worker.VERSION,status='inference_exported_matlab_pending',exports=[dict(file=q.name,sha256=worker.digest(q),input=i.name,input_sha256=worker.digest(i))])
  p=self.root/'records/a.json';p.write_text(json.dumps(record));self.assertTrue(worker.reusable(p,self.root))
  q.write_bytes(b'partial');self.assertFalse(worker.reusable(p,self.root))
 def test_stale_or_partial_matlab_result_is_pending(self):
  (self.root/'queue/a.mat').touch();(self.root/'records/a.json').write_text(json.dumps(dict(exports=[dict(file='a.mat',contract_id='current',scenes=20)])))
  p=self.root/'done/a.mat.json';p.write_text(json.dumps(dict(contract_id='old',rows=[{}]*20)))
  a=SimpleNamespace(work=self.root);self.assertEqual(controller.pending(a),1)
  p.write_text(json.dumps(dict(contract_id='current',rows=[{}]*19)));self.assertEqual(controller.pending(a),1)
  p.write_text(json.dumps(dict(contract_id='current',rows=[{}]*20)));self.assertEqual(controller.pending(a),0)
if __name__=='__main__':unittest.main()
