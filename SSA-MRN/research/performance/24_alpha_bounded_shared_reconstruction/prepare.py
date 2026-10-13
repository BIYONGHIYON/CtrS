"""Verify the one-cell contract and audit actual exports; never start QB training."""
import argparse
import ast
import copy
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2]
spec = importlib.util.spec_from_file_location("alpha_cell_under_test", HERE / "run.py")
cell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cell)


class ContractTests(unittest.TestCase):
    def test_cell_and_all_embedded_modules_compile(self):
        compile((HERE / "run.py").read_text(encoding="utf-8"), "cell", "exec")
        for source in (cell.MODEL, cell.SUPPORT, cell.WORKER, cell.INITIALIZATION):
            compile(source, "embedded", "exec")
        tree = ast.parse(cell.WORKER)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", cell.WORKER)
        self.assertNotIn("DataLoader", cell.WORKER)
        self.assertNotIn("clip_grad", cell.WORKER)
        # Final-output MSE, explicit fixed batch, FP32 validation/test.
        self.assertIn("F.mse_loss(pred.float(), b[\"gt\"])", cell.WORKER)
        self.assertIn("range(start, 100)", cell.WORKER)
        self.assertIn("order[j*32:(j+1)*32]", cell.WORKER)
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == "evaluate" for n in tree.body))

    def test_absent_resume_fails_before_any_download_or_gpu_access(self):
        with patch.object(cell, "REQUIRE_RESUME", True), patch.object(cell, "RESUME_INPUT", ""):
            with self.assertRaisesRegex(ValueError, "exact RESUME_INPUT"):
                cell.check_resume_input()
        with tempfile.TemporaryDirectory() as tmp, patch.object(cell, "REQUIRE_RESUME", True), patch.object(cell, "RESUME_INPUT", tmp):
            with self.assertRaises(FileNotFoundError): cell.check_resume_input()

    def test_resume_path_with_fresh_flag_cannot_silently_restart(self):
        with patch.object(cell, "REQUIRE_RESUME", False), patch.object(cell, "RESUME_INPUT", "/old"):
            with self.assertRaisesRegex(ValueError, "accidental fresh"): cell.check_resume_input()

    def test_download_missing_only_and_resolves_nested_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); existing = root / "train.h5"; existing.write_bytes(b"train")
            nested = root / "download/nested"; nested.mkdir(parents=True)
            paths = dict(train=str(existing), val="/missing/val.h5", RR="/missing/rr.h5", FR="/missing/fr.h5")
            for name in ("val.h5", "rr.h5", "fr.h5"): (nested / name).write_bytes(b"fixture")
            calls = []
            def download(handle): calls.append(handle); return root / "download"
            result = cell.ensure_qb_data(paths, download)
            self.assertEqual(calls, ["biyonghiyon/ctrs-pan-sharpening-dataset"])
            self.assertEqual(result["train"], str(existing))
            self.assertTrue(all(Path(v).is_file() for v in result.values()))
            self.assertEqual(cell.ensure_qb_data(result, lambda _: self.fail("Unexpected redownload")), result)

    def test_download_failure_or_ambiguous_file_stops(self):
        paths = {s:"/missing/"+s+".h5" for s in ("train","val","RR","FR")}
        def broken(_): raise OSError("offline")
        with self.assertRaisesRegex(RuntimeError, "No training started"):
            cell.ensure_qb_data(paths, broken)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ("a","b"):
                (root/d).mkdir(); (root/d/"train.h5").write_bytes(b"duplicate")
            with self.assertRaises(FileNotFoundError): cell.ensure_qb_data(paths, lambda _: root)

    def test_protocol_is_fixed_and_selection_is_not_test_driven(self):
        p = cell.protocol({}, {}, {}, {})
        self.assertEqual((p["k"],p["seed"],p["epochs"],p["batch"],p["micro"],p["lr"]), (4,42,100,32,32,1e-4))
        self.assertEqual(p["selection"]["scene_ids"], [1,8,11,13,19])
        self.assertTrue(p["selection"]["selected_before_training"])
        self.assertEqual(p["optimizer"]["fused"], True)
        self.assertEqual(p["optimizer"]["foreach"], False)
        self.assertEqual(p["stages"], 3)
        self.assertIn("no test-based selection", p["evaluation"])
        cfg=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
        for key in ("k","seed","epochs","batch","micro","lr","optimizer","selection","alpha"):
            self.assertEqual(cfg[key],p[key])

    def test_display_uses_only_current_session_and_replaces_fields(self):
        import inspect
        source = inspect.getsource(cell.run_workers)
        self.assertIn('current[arm] = value', source)
        self.assertIn('value.get("session_nonce") == session', source)
        self.assertIn('display_id=True', source)
        self.assertNotIn("history.json", source)
        self.assertNotIn("glob(", source)


class ModelTests(unittest.TestCase):
    def pair(self):
        torch.manual_seed(42); left = models.build("B1")
        torch.manual_seed(42); right = models.build("B1_alpha_bounded")
        return left, right

    def inputs(self, side=16):
        gen = torch.Generator().manual_seed(123)
        return [torch.rand((1,c,s,s), generator=gen) for c,s in ((1,side),(4,side),(4,side//4))]

    def test_same_core_encoder_alpha_and_initial_output(self):
        left,right = self.pair()
        self.assertEqual(models.common_hash(left), models.common_hash(right))
        for name,v in left.state_dict().items():
            if name.startswith(("core.","error_encoder.")): self.assertTrue(torch.equal(v,right.state_dict()[name]))
        self.assertAlmostEqual(float(left.effective_alpha().detach()), .1, delta=1e-7)
        self.assertAlmostEqual(float(right.effective_alpha().detach()), .1, delta=1e-7)
        self.assertAlmostEqual(float(right.a.detach()), math.log(.1/.9), delta=1e-6)
        with torch.inference_mode():
            l = left(*self.inputs(64)); r = right(*self.inputs(64))
        torch.testing.assert_close(l,r,atol=1e-6,rtol=1e-5)
        OBSERVED.update(common_initial_sha256=models.common_hash(left),
                        core_sha256=models.state_hash(left.core.state_dict()),
                        encoder_sha256=models.state_hash(left.error_encoder.state_dict()),
                        initial_alpha=dict(B1=float(left.effective_alpha().detach()),
                                           B1_alpha_bounded=float(right.effective_alpha().detach())),
                        synthetic_64px_initial_output_max_abs=float((l-r).abs().max()))

    def test_original_B1_weights_and_forward_are_unchanged(self):
        legacy_file = RESEARCH / "research/performance/15_shared_reconstruction/provenance/code/controlled_models.py"
        spec = importlib.util.spec_from_file_location("legacy_B1_read_only", legacy_file)
        legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
        torch.manual_seed(42); old = legacy.build("B1")
        torch.manual_seed(42); new = models.build("B1")
        self.assertEqual(models.state_hash(old.state_dict()), models.state_hash(new.state_dict()))
        with torch.inference_mode():
            torch.testing.assert_close(old(*self.inputs()),new(*self.inputs()),atol=0,rtol=0)
        OBSERVED["baseline_matches_archived_K4_model_state_and_forward"] = True

    def test_shared_modules_called_three_times_with_zero_error(self):
        for arm in cell.ARMS:
            model = models.build(arm); counts = dict(core=0,encoder=0)
            def core_hook(module,args,out): counts["core"] += 1
            def encoder_hook(module,args,out):
                counts["encoder"] += 1
                self.assertEqual(int(torch.count_nonzero(args[0])),0)
            h1=model.core.register_forward_hook(core_hook); h2=model.error_encoder.register_forward_hook(encoder_hook)
            with torch.inference_mode(): model(*self.inputs())
            h1.remove(); h2.remove()
            self.assertEqual(counts,dict(core=3,encoder=3))
            scalar = [n for n,_ in model.named_parameters() if n in ("alpha","a")]
            self.assertEqual(scalar,["a" if arm==cell.ARMS[1] else "alpha"])

    def test_zero_error_keeps_trainable_bias_and_sigmoid_chain_rule(self):
        left,right = self.pair(); inputs = self.inputs()
        for model in (left,right): model(*inputs).square().mean().backward()
        self.assertGreater(float(left.error_encoder.bias.grad.abs().sum()),0)
        self.assertEqual(float(left.error_encoder.weight.grad.abs().sum()),0)
        self.assertEqual(float(right.error_encoder.weight.grad.abs().sum()),0)
        torch.testing.assert_close(left.error_encoder.bias.grad,right.error_encoder.bias.grad,atol=1e-6,rtol=1e-4)
        alpha = right.effective_alpha().detach()
        torch.testing.assert_close(right.a.grad,left.alpha.grad*alpha*(1-alpha),atol=1e-6,rtol=1e-4)

    def test_bound_and_unbounded_behaviors_are_distinct(self):
        left,right = self.pair()
        with torch.no_grad(): left.alpha.fill_(1.5); right.a.fill_(8.)
        self.assertGreater(float(left.effective_alpha().detach()),1)
        self.assertGreater(float(right.effective_alpha().detach()),0)
        self.assertLess(float(right.effective_alpha().detach()),1)

    def test_initialization_subprocess_ignores_old_notebook_module_on_repeat(self):
        import h5py
        import types
        inputs=self.inputs(64)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"provenance").mkdir(); val=root/"val.h5"
            with h5py.File(val,"w") as f:
                for k,v in zip(("pan","lms","ms"),inputs): f[k]=v.repeat(2,1,1,1).numpy()*2047.
            stale=types.ModuleType("alpha_models")
            with patch.dict(sys.modules, {"alpha_models":stale}):
                for _ in range(2):
                    proof=cell.initialize_pair(CODE,root,val,False)
                    self.assertLessEqual(proof["output_max_abs"],1e-6)
                    self.assertIs(sys.modules["alpha_models"],stale)


class ResumeTests(unittest.TestCase):
    def fixture(self,arm):
        # CPU fixtures exercise state/replay only; these are not QB experiment metrics.
        torch.manual_seed(42); random.seed(42); np.random.seed(42)
        model = models.build(arm)
        class TinyCore(torch.nn.Module):
            def __init__(self): super().__init__(); self.conv=torch.nn.Conv2d(4,4,1)
            def forward(self,pan,lms,ms): return self.conv(lms)+pan*.01
        model.core = TinyCore()
        opt = torch.optim.Adam(model.parameters(),lr=1e-4,foreach=False)
        scaler = torch.amp.GradScaler("cpu",enabled=True)
        gen = torch.Generator().manual_seed(42); order = torch.randperm(5,generator=gen)
        p = dict(epochs=2,batch=2,data=dict(train=dict(shapes=dict(gt=[5,4,8,8]))))
        inputs_gen = torch.Generator().manual_seed(91)
        data = dict(pan=torch.rand((5,1,8,8),generator=inputs_gen),lms=torch.rand((5,4,8,8),generator=inputs_gen),
                    ms=torch.rand((5,4,2,2),generator=inputs_gen),gt=torch.rand((5,4,8,8),generator=inputs_gen))
        def step(ids):
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(*(data[k][ids] for k in ("pan","lms","ms"))),data["gt"][ids])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            return loss.detach()*data["gt"][ids].numel()
        sse=step(order[:2]); nr=np.random.get_state()
        ck=dict(protocol=p,arm=arm,epoch=0,best=None,best_epoch=0,history=[],model=model.state_dict(),
                optimizer=opt.state_dict(),scaler=scaler.state_dict(),cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.get_rng_state(),order_rng=gen.get_state(),python_rng=random.getstate(),
                numpy_rng=[nr[0],nr[1].tolist(),*nr[2:]],
                mid=dict(order=order,batch=1,count=2*4*8*8,sse=sse,seconds=.1,order_sha256=support.order_hash(order)))
        return model,opt,scaler,gen,step,ck,p

    def test_batch_replay_model_optimizer_scaler_rng_and_next_order(self):
        for arm in cell.ARMS:
            model,opt,scaler,gen,step,ck,p=self.fixture(arm)
            with tempfile.TemporaryDirectory() as tmp:
                record=support.commit_checkpoint(Path(tmp),ck,None)
                restored,_,_=support.load_record(Path(tmp),p,arm)
                sse=ck["mid"]["sse"].clone()
                for ids in (ck["mid"]["order"][2:4],ck["mid"]["order"][4:]): sse+=step(ids)
                expected=support.cpu(model.state_dict()); opt_expected=support.cpu(opt.state_dict())
                scaler_expected=scaler.state_dict(); next_order=torch.randperm(5,generator=gen)
                expected_draws=(torch.rand(3),random.random(),np.random.random())
                model.load_state_dict(restored["model"]); opt.load_state_dict(restored["optimizer"]); scaler.load_state_dict(restored["scaler"])
                gen.set_state(restored["order_rng"]); torch.set_rng_state(restored["cpu_rng"])
                random.setstate(restored["python_rng"]); nr=restored["numpy_rng"]
                np.random.set_state((nr[0],np.array(nr[1],dtype=np.uint32),*nr[2:]))
                actual_sse=restored["mid"]["sse"].clone()
                for ids in (restored["mid"]["order"][2:4],restored["mid"]["order"][4:]): actual_sse+=step(ids)
                for name,v in model.state_dict().items(): torch.testing.assert_close(v,expected[name],atol=0,rtol=0)
                for param,state in opt.state_dict()["state"].items():
                    for key,v in state.items(): torch.testing.assert_close(v,opt_expected["state"][param][key],atol=0,rtol=0)
                self.assertEqual(scaler.state_dict(),scaler_expected)
                torch.testing.assert_close(sse,actual_sse,atol=0,rtol=0)
                self.assertTrue(torch.equal(next_order,torch.randperm(5,generator=gen)))
                torch.testing.assert_close(expected_draws[0],torch.rand(3),atol=0,rtol=0)
                self.assertEqual(expected_draws[1],random.random()); self.assertEqual(expected_draws[2],np.random.random())

    def test_best_latest_transaction_survives_crash_before_exports(self):
        *_,ck,p=self.fixture("B1")
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); first=support.commit_checkpoint(folder,ck,None)
            end=support.cpu(ck); end.update(epoch=1,mid=None,best=.5,best_epoch=1)
            end["history"]=[dict(epoch=1,train_mse=.6,val_mse=.5,val_psnr=3.,val_sam=1.,alpha=.1,alpha_raw=.1,
                                  train_seconds=1.,validation_seconds=1.,sample_order_sha256="a"*64)]
            with patch.object(support,"export_record",side_effect=OSError("simulated export crash")):
                with self.assertRaises(OSError): support.commit_checkpoint(folder,end,first,new_best=True)
            recovered,record,origin=support.load_record(folder,p,"B1")
            self.assertEqual((recovered["epoch"],origin),(1,"resume.json"))
            support.export_record(folder,record)
            self.assertEqual(support.sha(folder/"best.pt"),support.sha(folder/"latest.pt"))

    def test_corrupt_latest_falls_back_to_matching_previous_transaction(self):
        *_,ck,p=self.fixture("B1")
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); first=support.commit_checkpoint(folder,ck,None)
            second=support.commit_checkpoint(folder,ck,first)
            (folder/second["latest"]["file"]).write_bytes(b"corrupt")
            recovered,record,origin=support.load_record(folder,p,"B1")
            self.assertEqual(origin,"resume.prev.json"); self.assertEqual(record,first)
            (folder/first["latest"]["file"]).write_bytes(b"also corrupt")
            with self.assertRaisesRegex(RuntimeError,"no fresh start"): support.load_record(folder,p,"B1")

    def test_invalid_cursor_order_rng_protocol_or_best_is_rejected(self):
        *_,ck,p=self.fixture("B1")
        changes=[lambda x:x["mid"].update(count=1), lambda x:x["mid"].update(batch=4),
                 lambda x:x["mid"].update(order=torch.zeros(5,dtype=torch.int64)),
                 lambda x:x.pop("order_rng"), lambda x:x.update(arm="B1_alpha_bounded"),
                 lambda x:x.update(epoch=3), lambda x:x.update(best=.1,best_epoch=1)]
        for change in changes:
            bad=copy.deepcopy(ck); change(bad)
            with self.assertRaises((ValueError,KeyError)): support.validate_state(bad,p,"B1")

    def test_canonical_file_damage_does_not_destroy_immutable_original(self):
        *_,ck,p=self.fixture("B1")
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); record=support.commit_checkpoint(folder,ck,None)
            (folder/"latest.pt").write_bytes(b"damaged export")
            support.load_record(folder,p,"B1"); support.export_record(folder,record)
            self.assertEqual(support.sha(folder/"latest.pt"),record["latest"]["sha256"])

    def test_crash_before_pointer_commit_keeps_old_state(self):
        *_,ck,p=self.fixture("B1")
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); first=support.commit_checkpoint(folder,ck,None)
            replace=support.os.replace
            def fail_pointer(source,target):
                if Path(target).name=="resume.json": raise OSError("simulated pointer crash")
                return replace(source,target)
            with patch.object(support.os,"replace",side_effect=fail_pointer):
                with self.assertRaises(OSError): support.commit_checkpoint(folder,ck,first)
            _,record,_=support.load_record(folder,p,"B1")
            self.assertEqual(record,first)

    def test_pruning_preserves_current_previous_and_best_originals(self):
        *_,ck,p=self.fixture("B1")
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); record=None
            for _ in range(5): record=support.commit_checkpoint(folder,ck,record)
            self.assertEqual(len(list((folder/"recovery").glob("*.pt"))),2)
            previous=json.loads((folder/"resume.prev.json").read_text(encoding='utf-8'))
            for rec in (record,previous): self.assertTrue(support.reference(folder,rec["latest"]).is_file())

    def test_result_export_crc_hashes_panels_curves_and_signed_differences(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"synthetic_export_fixture"; out.mkdir()
            p=None
            for arm in cell.ARMS:
                *_,ck,p_arm=self.fixture(arm)
                p=dict(p_arm,epochs=100,selection=dict(scene_ids=[1,8,11,13,19]))
                alpha=float(ck["model"]["alpha"] if arm=="B1" else ck["model"]["a"].sigmoid())
                raw=float(ck["model"]["alpha"] if arm=="B1" else ck["model"]["a"])
                h=[dict(epoch=e,train_mse=.6,val_mse=.5,val_psnr=3.,val_sam=1.,alpha=alpha,alpha_raw=raw,
                        train_seconds=1.,validation_seconds=1.,sample_order_sha256="a"*64) for e in range(1,101)]
                best=support.cpu(ck); best.update(protocol=p,epoch=1,mid=None,best=.5,best_epoch=1,history=h[:1])
                ckdir=out/"checkpoints"/arm
                record=support.commit_checkpoint(ckdir,best,None,new_best=True)
                latest=support.cpu(best); latest.update(epoch=100,history=h)
                support.commit_checkpoint(ckdir,latest,record)
                dest=out/"results"/arm; dest.mkdir(parents=True)
                result=dict(arm=arm,best_epoch=1,best_validation_mse=.5,best_alpha=alpha,weights={},images=[],metrics={})
                for kind,e in (("best",1),("latest",100)):
                    f=ckdir/(kind+".pt")
                    result["weights"][kind]=dict(file=str(f.relative_to(out)),epoch=e,bytes=f.stat().st_size,sha256=support.sha(f))
                for i in p["selection"]["scene_ids"]:
                    name=f"RR_scene_{i:02}.png"; Image.new("RGB",(1024,296),"white").save(dest/name)
                    result["images"].append(dict(scene_id=i,file=name))
                for split,metrics in (("RR",dict(PSNR=30.,SAM=2.,MSE_peak1=.01)),("FR",dict(D_lambda=.1,D_s=.2,QNR=.72))):
                    rows=[dict(scene_id=i,**metrics) for i in range(20)]
                    cell.write(dest/(split+"_per_scene.json"),rows)
                    result["metrics"][split]=dict(count=20,mean=metrics,rows=rows)
                cell.write(dest/"history.json",h); cell.write(dest/"results.json",result)
            archive,decision=cell.finish(out,p,support)
            self.assertEqual(decision,"개선 없음")
            manifest=json.loads((out/"manifest.json").read_text(encoding='utf-8'))
            for name,e in manifest["files"].items():
                self.assertEqual(support.sha(out/name),e["sha256"])
                self.assertEqual((out/name).stat().st_size,e["bytes"])
            checksum=json.loads(archive.with_suffix(".sha256.json").read_text(encoding='utf-8'))
            self.assertEqual(support.sha(archive),checksum["sha256"])
            self.assertFalse(checksum["remote_recovery_verified"])
            for name in ("learning.png","alpha.png","test_comparison.png"):
                with Image.open(out/"results"/name) as im: im.verify()
            self.assertEqual(len([k for k in manifest["files"] if "RR_scene_" in k]),10)


def audit_results(source, output):
    """Read-only audit of an actual export; synthetic forward is a load check, not test reevaluation."""
    import csv
    import math
    import numpy as np
    import torch
    import zipfile
    from PIL import Image
    source=Path(source).resolve()
    if not (source/'manifest.json').is_file():
        candidates=[p.parent for p in source.glob('*/manifest.json')]
        if len(candidates)!=1:raise ValueError('Specify the exact export folder or its single enclosing folder')
        source=candidates[0]
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for rel,expected in manifest['files'].items():
        path=(source/rel).resolve()
        if not path.is_relative_to(source):raise ValueError('Unsafe manifest path')
        if not path.is_file() or path.stat().st_size!=expected['bytes'] or cell.sha(path)!=expected['sha256']:
            raise ValueError('Original export hash mismatch: '+rel)
    p=json.loads((source/'provenance/protocol.json').read_text(encoding='utf-8'))
    declared=cell.protocol({}, {}, {}, {})
    for key in declared:
        if key not in ('data','sources','initialization','environment') and p[key]!=declared[key]:
            raise ValueError('Fixed protocol differs: '+key)
    for split,expected in cell.EXPECTED_DATA.items():
        if p['data'][split]['sha256']!=expected:raise ValueError('Recorded H5 hash differs: '+split)
    code=source/'provenance/code'
    expected_names=set(cell.EXPECTED_SOURCE)|{'alpha_models.py','alpha_support.py','alpha_worker.py','alpha_init.py'}
    if set(p['sources'])!=expected_names:raise ValueError('Unexpected source file inventory')
    for rel,digest in p['sources'].items():
        if cell.sha(code/rel)!=digest:raise ValueError('Executed source hash differs: '+rel)
    for rel,digest in cell.EXPECTED_SOURCE.items():
        if cell.sha(code/rel)!=digest:raise ValueError('Pinned core/metrics changed: '+rel)
    for name,text in [('alpha_models.py',cell.MODEL),('alpha_support.py',cell.SUPPORT),
                      ('alpha_worker.py',cell.WORKER),('alpha_init.py',cell.INITIALIZATION)]:
        if (code/name).read_bytes()!=text.encode('utf-8'):raise ValueError('Executed source differs from reviewed design: '+name)
    launcher=ast.parse((source/'provenance/run_cell.py').read_text(encoding='utf-8'))
    reviewed_launcher=ast.parse((HERE/'run.py').read_text(encoding='utf-8'))
    if ast.dump(launcher,include_attributes=False)!=ast.dump(reviewed_launcher,include_attributes=False):
        raise ValueError('Executed full cell differs from reviewed source')
    constants={n.targets[0].id:n.value.value for n in launcher.body if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    for key in ('MODEL','SUPPORT','WORKER','INITIALIZATION'):
        if constants[key]!=getattr(cell,key):raise ValueError('Cell source differs: '+key)
    launch=json.loads((source/'provenance/launch.json').read_text(encoding='utf-8'))
    if launch['resume'] or constants['REQUIRE_RESUME']:raise ValueError('This report requires the declared fresh paired run')
    # Execute only source already byte-verified against the reviewed model and pinned upstream.
    sys.path.insert(0,str(code/'SSA-MRN/src')); sys.path.insert(0,str(code))
    import alpha_models as audited_models
    audited_support=cell.load_support(code)
    init=torch.load(source/'provenance/initial_states.pt',map_location='cpu',weights_only=True)
    init_json=json.loads((source/'provenance/initialization_pair.json').read_text(encoding='utf-8'))
    if init_json!=p['initialization'] or cell.sha(source/'provenance/initial_states.pt')!=init_json['initial_states_sha256']:
        raise ValueError('Initialization evidence inconsistent')
    for arm,state in init['states'].items():
        common={k:v for k,v in state.items() if k.startswith(('core.','error_encoder.'))}
        if audited_models.state_hash(common)!=init_json['common_sha256']:raise ValueError('Initial common state differs')
        alpha=state['alpha'] if arm=='B1' else state['a'].sigmoid()
        if abs(float(alpha)-.1)>1e-7:raise ValueError('Initial alpha differs')
    gpu={a:torch.load(source/'provenance'/(a+'_initial_gpu.pt'),map_location='cpu',weights_only=True) for a in cell.ARMS}
    gpu_json=json.loads((source/'provenance/initialization_gpu_pair.json').read_text(encoding='utf-8'))
    for value in gpu.values():
        if value['common_sha256']!=init_json['common_sha256'] or abs(value['alpha']-.1)>1e-7:
            raise ValueError('Initial GPU common state/alpha differs')
        if not torch.isfinite(value['fp32']).all() or not torch.isfinite(value['amp']).all():
            raise ValueError('Nonfinite initial GPU output')
    for precision,atol,rtol in [('fp32',1e-6,1e-5),('amp',1e-3,1e-3)]:
        left,right=[gpu[a][precision] for a in cell.ARMS]
        torch.testing.assert_close(left,right,atol=atol,rtol=rtol)
        if float((left-right).abs().max())!=gpu_json[precision+'_max_abs']:raise ValueError('GPU output proof differs')
    results=json.loads((source/'results/results.json').read_text(encoding='utf-8')); compact=json.loads((source/'results/compact_epochs.json').read_text(encoding='utf-8'))
    selection=json.loads((source/'results/selection.json').read_text(encoding='utf-8'))
    if selection!=p['selection']:raise ValueError('Selection changed')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    arms={}; latest_states={}; histories={}; checked_images=0
    for arm in cell.ARMS:
        cfg=json.loads((source/'provenance'/(arm+'_config.json')).read_text(encoding='utf-8'))
        if cfg['protocol']!=p or cfg['resume'] or cfg['physical_gpu']!=cell.ARMS.index(arm):raise ValueError('Arm configuration differs')
        environment=json.loads((source/'provenance'/(arm+'_environment.json')).read_text(encoding='utf-8'))
        if environment['torch']!=p['environment']['torch'] or environment['cuda']!=p['environment']['cuda'] or not environment['deterministic'] or environment['tf32']:
            raise ValueError('Training environment flags differ')
        ckdir=source/'checkpoints'/arm
        latest,record,pointer=audited_support.load_record(ckdir,p,arm)
        for pointer_name in ('resume.json','resume.prev.json'):
            saved=json.loads((ckdir/pointer_name).read_text(encoding='utf-8'))
            for ref in saved.values():
                if ref is not None:
                    path=audited_support.reference(ckdir,ref)
                    state=torch.load(path,map_location='cpu',weights_only=True)
                    audited_support.validate_state(state,p,arm)
                    with zipfile.ZipFile(path) as z:
                        if z.testzip() is not None:raise ValueError('Original checkpoint CRC differs')
        if latest['epoch']!=100 or latest['mid'] is not None:raise ValueError('Incomplete 100-epoch training')
        h=json.loads((source/'results'/arm/'history.json').read_text(encoding='utf-8'))
        if h!=latest['history'] or h!=compact[arm]:raise ValueError('Measured history differs')
        histories[arm]=h; latest_states[arm]=latest['model']
        with (source/'results'/arm/'epochs.csv').open(newline='') as f:
            csv_rows=list(csv.DictReader(f))
        if len(csv_rows)!=100:raise ValueError('CSV incomplete')
        for row,csv_row in zip(h,csv_rows):
            for key,value in row.items():
                if isinstance(value,str):
                    if csv_row[key]!=value:raise ValueError('CSV order hash differs')
                elif not math.isclose(float(csv_row[key]),value,rel_tol=1e-12,abs_tol=1e-15):raise ValueError('CSV differs')
            effective=row['alpha_raw'] if arm=='B1' else 1/(1+math.exp(-row['alpha_raw']))
            if not math.isclose(effective,row['alpha'],rel_tol=1e-6,abs_tol=1e-7):raise ValueError('Effective alpha curve differs')
        r=results[arm]
        if r!=json.loads((source/'results'/arm/'results.json').read_text(encoding='utf-8')):raise ValueError('Per-arm result differs')
        expected_best=min(h,key=lambda row:row['val_mse'])
        if (r['best_epoch'],r['best_validation_mse'])!=(expected_best['epoch'],expected_best['val_mse']):raise ValueError('Best selection differs')
        weight_info={}
        for kind in ('best','latest'):
            path=ckdir/(kind+'.pt'); ck=torch.load(path,map_location='cpu',weights_only=True)
            audited_support.validate_state(ck,p,arm)
            w=r['weights'][kind]
            if cell.sha(path)!=w['sha256'] or ck['epoch']!=w['epoch'] or path.stat().st_size!=w['bytes']:
                raise ValueError('Original checkpoint differs')
            if cell.sha(path)!=record[kind]['sha256']:raise ValueError('Checkpoint export differs from committed transaction')
            model=audited_models.build(arm).eval(); model.load_state_dict(ck['model'],strict=True)
            alpha=float(model.effective_alpha().detach())
            if not math.isclose(alpha,h[ck['epoch']-1]['alpha'],rel_tol=0,abs_tol=1e-7):raise ValueError('Weight alpha differs')
            gen=torch.Generator().manual_seed(7)
            inputs=[torch.rand((1,c,s,s),generator=gen) for c,s in ((1,16),(4,16),(4,4))]
            with torch.inference_mode(): prediction=model(*inputs)
            if prediction.shape!=(1,4,16,16) or not torch.isfinite(prediction).all():raise ValueError('Strict CPU inference failed')
            weight_info[kind]=dict(epoch=ck['epoch'],sha256=cell.sha(path),bytes=path.stat().st_size,alpha=alpha,
                                   model_state_sha256=audited_models.state_hash(ck['model']))
        for split in ('RR','FR'):
            rows=r['metrics'][split]['rows']
            if r['metrics'][split]['count']!=20 or [x['scene_id'] for x in rows]!=list(range(20)):
                raise ValueError('Full test scene set differs')
            if rows!=json.loads((source/'results'/arm/(split+'_per_scene.json')).read_text(encoding='utf-8')):raise ValueError('Scene JSON differs')
            for key,value in r['metrics'][split]['mean'].items():
                mean=float(np.mean([x[key] for x in rows]))
                if not all(math.isfinite(x[key]) for x in rows) or not math.isclose(mean,value,rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Scene metric mean differs')
        if [x['scene_id'] for x in r['images']]!=selection['scene_ids']:raise ValueError('Five scene selection differs')
        for item in r['images']:
            path=source/'results'/arm/item['file']
            with Image.open(path) as im:im.verify()
            with Image.open(path) as im:
                if im.size!=(1024,296):raise ValueError('Four-panel size differs')
            checked_images+=1
        alphas=[x['alpha'] for x in h]
        arms[arm]=dict(best_epoch=r['best_epoch'],best_validation_mse=r['best_validation_mse'],weights=weight_info,
          epoch_alpha=dict(minimum=min(alphas),minimum_epoch=1+int(np.argmin(alphas)),maximum=max(alphas),maximum_epoch=1+int(np.argmax(alphas)),
                           latest=alphas[-1],out_of_unit_interval_count=sum(x<0 or x>1 for x in alphas),sampling='epoch end only; intra-batch extrema unavailable'),
          train_minutes=sum(x['train_seconds'] for x in h)/60,validation_minutes=sum(x['validation_seconds'] for x in h)/60)
    if [x['sample_order_sha256'] for x in histories['B1']]!=[x['sample_order_sha256'] for x in histories['B1_alpha_bounded']]:
        raise ValueError('Paired sample orders differ')
    delta={s:{k:results['B1_alpha_bounded']['metrics'][s]['mean'][k]-v for k,v in results['B1']['metrics'][s]['mean'].items()} for s in ('RR','FR')}
    comparison=json.loads((source/'results/paired_comparison.json').read_text(encoding='utf-8'))
    if delta!=comparison['delta_candidate_minus_B1']:raise ValueError('Paired mean delta differs')
    for name in ('learning.png','alpha.png','test_comparison.png'):
        with Image.open(source/'results'/name) as im:im.verify()
    counts={}
    for split in ('RR','FR'):
        counts[split]={}
        for key in delta[split]:
            sign=1 if key in ('PSNR','SCC','Q2n','QNR') else -1
            differences=[results['B1_alpha_bounded']['metrics'][split]['rows'][i][key]-results['B1']['metrics'][split]['rows'][i][key] for i in range(20)]
            for i,value in enumerate(differences):
                if value!=comparison['per_scene_delta'][split][i][key]:raise ValueError('Paired scene delta differs')
            counts[split][key]=dict(better=sum(sign*x>0 for x in differences),worse=sum(sign*x<0 for x in differences),equal=sum(x==0 for x in differences))
    previous=RESEARCH/'research/performance/15_shared_reconstruction'
    previous_latest=torch.load(previous/'checkpoints/B1/latest.pt',map_location='cpu',weights_only=True)
    previous_best=torch.load(previous/'checkpoints/B1/best.pt',map_location='cpu',weights_only=True)
    current_best=torch.load(source/'checkpoints/B1/best.pt',map_location='cpu',weights_only=True)
    tensor_equal=lambda left,right:set(left)==set(right) and all(torch.equal(left[k],right[k]) for k in left)
    comparable=('train_mse','val_mse','val_psnr','val_sam','sample_order_sha256')
    directions=[('RR','PSNR',1),('RR','SAM',-1),('RR','MSE_peak1',-1),('FR','D_lambda',-1),('FR','D_s',-1),('FR','QNR',1)]
    good=[sign*delta[split][key] for split,key,sign in directions]
    rr_preserved=all(x>=0 for x in good[:3]);fr_reduced=all(x>=0 for x in good[3:]) and any(x>0 for x in good[3:])
    decision='개선' if rr_preserved and fr_reduced else '부분 개선' if any(x>0 for x in good) else '개선 없음'
    if comparison['decision']!=decision or comparison['rr_non_degraded_in_this_seed']!=rr_preserved or comparison['fr_distortion_reduced_in_this_seed']!=fr_reduced:raise ValueError('Reported decision differs')
    preceding=dict(current_B1_best_tensors_equal=tensor_equal(current_best['model'],previous_best['model']),
                   current_B1_latest_tensors_equal=tensor_equal(latest_states['B1'],previous_latest['model']),
                   all_100_epoch_values_equal=all(all(x[k]==y[k] for k in comparable) for x,y in zip(histories['B1'],previous_latest['history'])),
                   old_checkpoint_bytes_preserved=True)
    report=dict(status='passed',source_run=source.name,source_manifest_files_checked=len(manifest['files']),
                source_manifest_sha256=cell.sha(source/'manifest.json'),source_bytes=sum(x['bytes'] for x in manifest['files'].values()),
                reviewed_source_bytes_equal=True,full_executed_cell_AST_equal=True,original_checkpoint_ZIP_CRC_verified=True,
                recorded_H5_hashes_match=True,actual_H5_files_rehashed=False,
                fresh_paired_run_declared=True,initial_core_encoder_equal=True,saved_initial_GPU_outputs_equal=True,
                all_100_epoch_orders_equal=True,earliest_validation_MSE_best_verified=True,checkpoint_history_curve_csv_equal=True,
                strict_CPU_load_and_finite_forward_count=4,synthetic_forward_is_not_test_reevaluation=True,
                independent_test_scenes_per_arm=dict(RR=20,FR=20),RR_four_panel_images_checked=checked_images,
                arms=arms,delta_candidate_minus_B1=delta,scene_comparison_counts=counts,previous_B1_comparison=preceding,
                decision=decision,rr_preservation_target_met=rr_preserved,FR_metrics_provisional=True,
                actual_H5_GT_metrics_recomputed=False,CUDA_training_or_replay_rerun=False,remote_archive_verified=False)
    cell.write(output,report)
    print(json.dumps(dict(status='passed',manifest_files=len(manifest['files']),strict_CPU_forwards=4,
                         fixed_images=checked_images,delta=delta,alpha={a:arms[a]['epoch_alpha'] for a in cell.ARMS},
                         scene_counts=counts,previous_B1=preceding),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--audit-results",type=Path,help="Read-only audit of an actual Kaggle export; never train")
    args=parser.parse_args()
    if args.audit_results:
        audit_results(args.audit_results,args.output or HERE/'results/verification.json');return
    if not args.verify: parser.print_help(); return
    global torch,np,models,support,OBSERVED,CODE
    import torch
    import numpy as np
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    OBSERVED={}
    with tempfile.TemporaryDirectory(prefix="ctrs_alpha_preflight_") as tmp:
        code=Path(tmp)/"code"; code.mkdir()
        CODE=code
        sources=cell.bootstrap_sources(code,False)
        sys.path.insert(0,str(code/"SSA-MRN/src")); sys.path.insert(0,str(code))
        import alpha_models as models
        support=cell.load_support(code)
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
        cell.write(args.output or HERE/'provenance/preflight.json',dict(status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
                    failures=len(result.failures),errors=len(result.errors),torch=str(torch.__version__),device="CPU",
                    evidence=OBSERVED,source_hashes=sources,
                    verified="syntax, archived B1 parity, initial outputs/gradients, download resolver, CPU replay and crash recovery",
                    not_verified=["actual QB training", "T4x2 GPU residency/AMP/fused optimizer", "Kaggle dataset download on live service",
                                  "actual full RR/FR evaluation", "Saved Output remote recovery"]))
        if not result.wasSuccessful(): sys.exit(1)


if __name__=="__main__": main()
