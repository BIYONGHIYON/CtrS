"""CPU/stdlib contract tests; these do NOT certify PyTorch/CUDA training."""
import ast
import copy
import pathlib
import tempfile
import types
import unittest
import os
import subprocess
import sys
import json

CELL = pathlib.Path(__file__).with_name("kaggle_b0_b1_k4_controlled_one_cell.py")
TREE = ast.parse(CELL.read_text(encoding="utf-8"))
STRINGS = {n.targets[0].id:n.value.value for n in TREE.body if isinstance(n,ast.Assign)
           and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}
WORKER = ast.parse(STRINGS["WORKER"])

class Order:
    dtype="int64"
    def __init__(self,values): self.values=values
    def sort(self): return types.SimpleNamespace(values=Order(sorted(self.values)))

class ContractTests(unittest.TestCase):
    def test_initialization_ignores_cached_notebook_module_on_reexecution(self):
        node=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=="verify_initialization")
        env=dict(os=os,subprocess=subprocess,sys=sys,json=json,Path=pathlib.Path,
                 INITIALIZATION_CHECK=STRINGS["INITIALIZATION_CHECK"])
        exec(compile(ast.Module(body=[node],type_ignores=[]),"initialization","exec"),env)
        stale=types.ModuleType("controlled_models")
        previous=sys.modules.get("controlled_models")
        sys.modules["controlled_models"]=stale
        try:
            for _ in range(2):
                with tempfile.TemporaryDirectory() as tmp:
                    root=pathlib.Path(tmp)
                    (root/"torch.py").write_text("def manual_seed(seed): pass\n",encoding="utf-8")
                    (root/"controlled_models.py").write_text(
                        "class Model:\n    def state_dict(self): return {}\n"
                        "def build(arm): return Model()\n"
                        "def core(model): return model\n"
                        "def state_hash(state): return 'a'*64\n",encoding="utf-8")
                    self.assertEqual(env["verify_initialization"](root),{"B0":"a"*64,"B1":"a"*64})
                    self.assertIs(sys.modules["controlled_models"],stale)
        finally:
            if previous is None: sys.modules.pop("controlled_models",None)
            else: sys.modules["controlled_models"]=previous

    def data_resolver(self):
        node=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=="ensure_qb_data")
        env=dict(Path=pathlib.Path)
        exec(compile(ast.Module(body=[node],type_ignores=[]),"data_resolver","exec"),env)
        return env["ensure_qb_data"]

    def test_existing_data_skips_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp)/"train.h5"; p.touch()
            def forbidden(handle): raise AssertionError("Unexpected download")
            self.assertEqual(self.data_resolver()({"train":str(p)},forbidden),{"train":str(p)})

    def test_missing_data_downloads_and_resolves_nested_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp); (root/"nested").mkdir(); (root/"nested/train.h5").touch()
            calls=[]
            def download(handle): calls.append(handle); return str(root)
            result=self.data_resolver()({"train":str(root/"absent/train.h5")},download)
            self.assertEqual(calls,["biyonghiyon/ctrs-pan-sharpening-dataset"])
            self.assertEqual(result["train"],str((root/"nested/train.h5").resolve()))

    def test_download_missing_required_file_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self.data_resolver()({"train":str(pathlib.Path(tmp)/"train.h5")},lambda handle:tmp)

    def test_download_failure_stops(self):
        def failed(handle): raise OSError("network unavailable")
        with self.assertRaisesRegex(RuntimeError,"학습은 시작하지 않았습니다"):
            self.data_resolver()({"train":"missing/train.h5"},failed)

    def test_embedded_syntax(self):
        for name in ("MODEL","WORKER","INITIALIZATION_CHECK"): compile(STRINGS[name],name,"exec")
        compile(CELL.read_text(encoding="utf-8"),str(CELL),"exec")

    def test_b1_definition(self):
        tree=ast.parse(STRINGS["MODEL"])
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="forward")
        loop=next(n for n in forward.body if isinstance(n,ast.For))
        self.assertEqual(ast.unparse(loop.iter),"range(3)")
        self.assertEqual([ast.unparse(n) for n in loop.body],[
            "conditioned = current + self.error_encoder(torch.zeros_like(current))",
            "proposal = self.core(pan, conditioned, ms)",
            "current = current + self.alpha * (proposal - current)"])
        self.assertNotIn("clamp",STRINGS["MODEL"])
        self.assertIn("bias=True",STRINGS["MODEL"])
        self.assertIn("RestoredPansharpeningNet(4, 4)",STRINGS["MODEL"])

    def harness(self,resume=True):
        state=dict(protocol={"id":"test"},arm="B0",epoch=0,history=[],mid=None,best_epoch=0,best=float("inf"))
        files={"latest.pt":state}
        def load(path,**kw):
            value=files[path.name]
            if isinstance(value,Exception): raise value
            return value
        torch=types.SimpleNamespace(load=load,int64="int64",arange=lambda n:Order(list(range(n))),
                                    equal=lambda a,b:a.values==b.values)
        env=dict(torch=torch,C=dict(resume=resume,protocol={"id":"test"},arm="B0",shapes={"train":{"gt":[64]}}),
                 RUN=pathlib.Path("unused"),math=__import__("math"),status=lambda *a,**k:None)
        node=next(n for n in WORKER.body if isinstance(n,ast.FunctionDef) and n.name=="recover")
        exec(compile(ast.Module(body=[node],type_ignores=[]),"recover","exec"),env)
        return env["recover"],files,state

    def test_fresh_only_when_explicit(self):
        recover,files,state=self.harness(False); files.clear(); self.assertIsNone(recover())
    def test_resume_missing_stops(self):
        recover,files,state=self.harness(); files.clear()
        with self.assertRaisesRegex(RuntimeError,"no fresh start"): recover()
    def test_corrupt_latest_falls_back(self):
        recover,files,state=self.harness(); files["latest.prev.pt"]=state; files["latest.pt"]=ValueError("corrupt")
        self.assertIs(recover(),state)
    def test_protocol_mismatch_stops(self):
        recover,files,state=self.harness(); state["protocol"]={"id":"old_k6"}
        with self.assertRaises(RuntimeError): recover()
    def test_completed_batches_preserved(self):
        recover,files,state=self.harness(); state["mid"]=dict(batch=2,order=Order(list(range(64))))
        self.assertEqual(recover()["mid"]["batch"],2)
        training=next(n for n in WORKER.body if isinstance(n,ast.FunctionDef) and n.name=="train")
        self.assertIn("range(cursor, nb)",ast.unparse(training))
    def test_bad_permutation_stops(self):
        recover,files,state=self.harness(); state["mid"]=dict(batch=1,order=Order([0]*64))
        with self.assertRaises(RuntimeError): recover()
    def test_epoch_overrun_stops(self):
        recover,files,state=self.harness(); state["epoch"]=101; state["history"]=[{}]*101
        with self.assertRaises(RuntimeError): recover()
    def test_best_transaction_stops(self):
        recover,files,state=self.harness(); state.update(epoch=1,history=[{}],best_epoch=1,best=.1)
        files["best.pt"]=dict(protocol=state["protocol"],epoch=2,best=.05)
        with self.assertRaises(RuntimeError): recover()

if __name__=="__main__": unittest.main()
