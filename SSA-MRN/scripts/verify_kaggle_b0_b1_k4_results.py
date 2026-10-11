"""Verify exported results without executing model code or arbitrary pickle globals.

Reads PyTorch ZIP storage with a restricted metadata reader and NumPy. This is
not torch.load, a model forward pass, or a fresh H5/GT metric computation.
"""
import argparse, ast, collections, hashlib, io, json, pickle, shutil, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""): h.update(b)
    return h.hexdigest()

def read_checkpoint(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        member=next(n for n in z.namelist() if n.endswith("/data.pkl"))
        prefix=member.rsplit("/",1)[0]
        byteorder=z.read(prefix+"/byteorder").decode()
        assert byteorder=="little"
        cache={}
        def rebuild(storage,offset,shape,stride,*unused):
            return np.ndarray(tuple(shape),dtype=storage.dtype,buffer=storage,
                              offset=offset*storage.dtype.itemsize,
                              strides=tuple(s*storage.dtype.itemsize for s in stride))
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if (module,name)==("collections","OrderedDict"): return collections.OrderedDict
                if (module,name)==("torch._utils","_rebuild_tensor_v2"): return rebuild
                if module=="torch" and name in ("FloatStorage","ByteStorage","LongStorage"):
                    return {"FloatStorage":np.dtype("<f4"),"ByteStorage":np.dtype("u1"),"LongStorage":np.dtype("<i8")}[name]
                raise ValueError(f"Disallowed pickle global {module}.{name}")
            def persistent_load(self,pid):
                kind,dtype,key,location,size=pid
                assert kind=="storage" and location=="cpu"
                if key not in cache:
                    cache[key]=np.frombuffer(z.read(prefix+"/data/"+key),dtype=dtype)
                    assert len(cache[key])==size
                return cache[key]
        return Reader(io.BytesIO(z.read(member))).load()

def j(path): return json.loads(path.read_text(encoding="utf-8"))
def write(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")

def verify(source,repo):
    manifest=j(source/"artifact_manifest.json")["artifacts"]
    for rel,info in manifest.items():
        p=(source/rel).resolve(); assert p.is_relative_to(source.resolve())
        assert p.stat().st_size==info["bytes"] and sha(p)==info["sha256"],rel
    protocol=j(source/"protocol.json"); results=j(source/"results.json")
    assert protocol["k"]==4 and protocol["epochs"]==100 and protocol["seed"]==42
    assert protocol["batch"]==protocol["micro"]==32 and protocol["lr"]==1e-4
    for rel,digest in protocol["sources"].items(): assert sha(source/"code"/rel)==digest
    for p in (source/"code").rglob("*.py"): ast.parse(p.read_text(encoding="utf-8"))
    selection=j(source/"selection.json")
    assert selection==protocol["selection"]
    assert selection["scene_ids"]==sorted(np.random.default_rng(42).choice(20,5,replace=False).tolist())
    init=j(source/"initialization_pair.json"); assert init["B0"]==init["B1"] and init["equal"]
    details={}; histories={}; checkpoints={}; predictions=0; panels=0
    for arm in ("B0","B1"):
        r=results[arm]; run=source/arm; history=j(run/"history.json"); histories[arm]=history
        assert len(history)==100 and [h["epoch"] for h in history]==list(range(1,101))
        best=min(history,key=lambda h:h["val_mse"])
        assert best["epoch"]==r["best_epoch"] and best["val_mse"]==r["best_validation_mse"]
        assert j(run/"initialization.json")["core_sha256"]==init[arm]
        assert r==j(run/"results.json") and j(source/(arm+"_config.json"))["resume"] is False
        checkpoints[arm]={}
        for kind in ("best","latest","latest.prev"):
            p=run/(kind+".pt"); ck=read_checkpoint(p)
            assert ck["protocol"]==protocol and ck["arm"]==arm
            assert ck["epoch"]==len(ck["history"])
            if kind in ("best","latest"): assert ck["mid"] is None
            elif ck["mid"]:
                n=protocol["data"]["train"]["shapes"]["gt"][0]
                assert 0<=ck["mid"]["batch"]<=(n+31)//32
                assert np.array_equal(np.sort(ck["mid"]["order"]),np.arange(n))
            assert ck["history"]==history[:ck["epoch"]]
            assert ck["best_epoch"]==r["best_epoch"] and ck["best"]==r["best_validation_mse"]
            if kind=="best": assert ck["epoch"]==r["best_epoch"]
            if kind=="latest": assert ck["epoch"]==100
            for tensor in ck["model"].values(): assert np.isfinite(tensor).all()
            for state in ck["optimizer"]["state"].values():
                for value in state.values():
                    if isinstance(value,np.ndarray): assert np.isfinite(value).all()
            group=ck["optimizer"]["param_groups"][0]
            assert group["lr"]==1e-4 and group["fused"] is True and group["weight_decay"]==0
            assert "cpu_rng" in ck and "cuda_rng" in ck and "order_rng" in ck and "numpy_rng" in ck
            assert len(ck["scaler"])>0
            alpha=float(ck["model"]["alpha"]) if arm=="B1" else None
            if arm=="B1":
                assert ck["model"]["error_encoder.weight"].shape==(4,4,3,3)
                assert ck["model"]["error_encoder.bias"].shape==(4,)
            info=dict(epoch=ck["epoch"],mid_epoch_batches=ck["mid"]["batch"] if ck["mid"] else None,
                      sha256=sha(p),bytes=p.stat().st_size,alpha=alpha,
                      model_tensor_count=len(ck["model"]),optimizer_state_count=len(ck["optimizer"]["state"]))
            if kind in ("best","latest"):
                expected=r["weights"][kind]
                assert all(info[k]==expected[k] for k in ("epoch","sha256","bytes"))
            checkpoints[arm][kind]=info
        for split in ("RR","FR"):
            rows=j(run/(split+"_per_scene.json")); assert rows==r["metrics"][split]["rows"]
            assert len(rows)==20 and [x["scene_id"] for x in rows]==list(range(20))
            for key,v in r["metrics"][split]["mean"].items():
                assert np.isclose(np.mean([x[key] for x in rows]),v,rtol=0,atol=1e-12)
            files=sorted((run/"predictions"/split).glob("*.npz")); assert len(files)==20
            shape=(4,256,256) if split=="RR" else (4,512,512)
            for p in files:
                with np.load(p,allow_pickle=False) as z:
                    a=z["prediction_dn"]; assert a.shape==shape and np.isfinite(a).all()
                predictions+=1
        assert sorted(x["scene_id"] for x in r["images"])==selection["scene_ids"]
        for item in r["images"]:
            with Image.open(run/item["file"]) as im:
                assert im.size==(1024,296); im.verify()
            panels+=1
        log=(run/"worker.log").read_text(encoding="utf-8")
        statuses=[]
        for line in log.splitlines():
            try: statuses.append(json.loads(line))
            except ValueError: pass
        assert not any(s.get("phase")=="resume" for s in statuses)
        assert any(s.get("phase")=="complete" for s in statuses)
        details[arm]=dict(best_epoch=r["best_epoch"],best_validation_mse=r["best_validation_mse"],
                         RR=r["metrics"]["RR"]["mean"],FR=r["metrics"]["FR"]["mean"],
                         train_minutes=sum(h["train_seconds"] for h in history)/60,
                         validation_minutes=sum(h["validation_seconds"] for h in history)/60,
                         RR_psnr_wins=None)
    assert [h["sample_order_sha256"] for h in histories["B0"]]==[h["sample_order_sha256"] for h in histories["B1"]]
    delta={s:{k:results["B1"]["metrics"][s]["mean"][k]-v for k,v in results["B0"]["metrics"][s]["mean"].items()} for s in ("RR","FR")}
    assert delta==j(source/"paired_comparison.json")["delta_B1_minus_B0"]
    wins=sum(b["PSNR"]>a["PSNR"] for a,b in zip(results["B0"]["metrics"]["RR"]["rows"],results["B1"]["metrics"]["RR"]["rows"]))
    oldroot=repo/"SSA-MRN/docs/assets/kaggle_b1_b2_100_resume"
    old=j(oldroot/"results.json"); olddata=j(oldroot/"data_manifest.json")
    data_equal={s:protocol["data"][s]["sha256"]==olddata[s]["sha256"] and protocol["data"][s]["shapes"]==olddata[s]["shapes"] for s in ("train","val","RR","FR")}
    assert all(data_equal.values())
    old_b1=old["B1_QB_B1_k6_s42"]
    oldhist=j(oldroot/"runs/B1_QB_B1_k6_s42/history.json")
    oldb0=j(repo/"SSA-MRN/docs/assets/kaggle_band_gated_hf/results.json")["github_QB_baseline_k6_s42"]
    cross={arm:{s:{k:results[arm]["metrics"][s]["mean"][k]-previous["metrics"][s]["mean"][k] for k in results[arm]["metrics"][s]["mean"]} for s in ("RR","FR")} for arm,previous in (("B0",oldb0),("B1",old_b1))}
    same_order=[h["sample_order_sha256"] for h in histories["B1"]]==[h["sample_order_sha256"] for h in oldhist]
    old_model=oldroot/"code/B2_models.py"
    old_tree=ast.parse(old_model.read_text(encoding="utf-8")); new_tree=ast.parse((source/"code/controlled_models.py").read_text(encoding="utf-8"))
    def updates(tree):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="SharedThreeStage")
        forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="forward")
        loop=next(n for n in forward.body if isinstance(n,ast.For))
        assert ast.unparse(loop.iter)=="range(3)"
        class ZeroError(ast.NodeTransformer):
            def visit_Call(self,node):
                node=self.generic_visit(node)
                if isinstance(node.func,ast.Attribute) and node.func.attr=="error_encoder" and ast.unparse(node.args[0])=="torch.zeros_like(current)":
                    node.args[0]=ast.Name(id="error",ctx=ast.Load())
                return node
        loop=ZeroError().visit(loop)
        return [ast.dump(n) for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("conditioned","proposal","current")]
    assert "error = torch.zeros_like(current)" in ast.unparse(old_tree)
    assert updates(old_tree)==updates(new_tree)
    code_equal={name:sha(source/"code"/name)==sha(oldroot/"code/CtrS"/name) for name in protocol["sources"] if name.startswith("SSA-MRN/")}
    assert all(code_equal.values())
    return dict(source_manifest_files=len(manifest),all_source_manifest_hashes_valid=True,
                predictions_checked=predictions,RR_panels_checked=panels,completed_epochs_per_arm=100,
                initial_core_sha256=init["B0"],within_K4_epoch_orders_equal=True,checkpoints=checkpoints,
                summary=details,delta_B1_minus_B0=delta,B1_RR_PSNR_scene_wins=wins,
                k6_comparison=dict(completely_identical_experiment=False,data_sha_and_shapes_equal=data_equal,
                    shared_B1_update_AST_equal=True,adapter_metrics_upstream_bytes_equal=code_equal,
                    K4_B1_vs_K6_B1_epoch_orders_equal=same_order,K4_minus_K6_observed_deltas=cross,
                    differences=["K4 versus K6","fresh controlled B0/B1 versus external B0 and resumed B1/B2",
                                 "B0 fused Adam versus historical foreach Adam","different full model/worker and checkpoint formats",
                                 "observation diagnostics removed from zero-error B1; RR/FR metric implementation unchanged"]),
                checkpoint_validation="Restricted PyTorch ZIP metadata + NumPy storage, CRC and finite tensors; not torch.load or forward",
                limitations=["No local training, model forward or original H5/GT metric recalculation",
                             "FR lacks GT; QNR/D_s and Q2n/SCC MATLAB parity limitations remain",
                             "Archive ZIP/SHA sidecar was not supplied; extracted artifact manifest checked instead"])

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("source",type=Path); parser.add_argument("--copy",action="store_true")
    args=parser.parse_args(); repo=Path(__file__).resolve().parents[2]
    proof=verify(args.source,repo)
    target=repo/"SSA-MRN/docs/assets/kaggle_b0_b1_k4_controlled"
    if args.copy:
        assert not target.exists(),"Refuse to overwrite archived evidence"
        shutil.copytree(args.source,target)
        write(target/"verification.json",proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2))
