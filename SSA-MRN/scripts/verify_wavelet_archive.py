"""Verify downloaded QB wavelet artifacts without CUDA or importing torch.

Reads only a small allowlist of tensor reconstruction globals in torch.save ZIPs.
This verifies stored weights/configs, not model inference or source H5 contents.
"""
import argparse
import ast
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import zipfile

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


class TensorReader(pickle.Unpickler):
    def __init__(self, archive, prefix, payload):
        super().__init__(io.BytesIO(payload)); self.archive=archive; self.prefix=prefix; self.storages={}

    def find_class(self, module, name):
        if (module, name)==("collections", "OrderedDict"): return OrderedDict
        if module=="torch" and name in ("FloatStorage","DoubleStorage","HalfStorage","ByteStorage","LongStorage","IntStorage","BoolStorage"):
            return np.dtype({"FloatStorage":"<f4","DoubleStorage":"<f8","HalfStorage":"<f2",
                             "ByteStorage":"u1","LongStorage":"<i8","IntStorage":"<i4","BoolStorage":"?"}[name])
        if module=="torch._utils" and name=="_rebuild_tensor_v2": return self.tensor
        raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")

    def persistent_load(self, pid):
        kind,dtype,key,device,count=pid
        if kind!="storage": raise pickle.UnpicklingError("Unknown persistent object")
        if key not in self.storages:
            data=self.archive.read(self.prefix+"data/"+key)
            if len(data)!=count*dtype.itemsize: raise ValueError("Storage byte count mismatch")
            self.storages[key]=np.frombuffer(data,dtype=dtype)
        return self.storages[key]

    @staticmethod
    def tensor(storage, offset, shape, strides, requires_grad, hooks, metadata=None):
        if len(shape)!=len(strides) or offset<0 or any(s<0 for s in strides): raise ValueError("Invalid tensor geometry")
        last=offset+sum((d-1)*s for d,s in zip(shape,strides)) if all(shape) else offset
        if last>=len(storage) and math.prod(shape): raise ValueError("Tensor exceeds storage")
        return np.ndarray(shape,dtype=storage.dtype,buffer=storage,offset=offset*storage.dtype.itemsize,
                          strides=tuple(s*storage.dtype.itemsize for s in strides))


def checkpoint(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None: raise ValueError(f"Checkpoint ZIP CRC failure: {path}")
        p=[n for n in z.namelist() if n.endswith("/data.pkl")]
        if len(p)!=1: raise ValueError("Expected one checkpoint pickle")
        prefix=p[0][:-len("data.pkl")]
        return TensorReader(z,prefix,z.read(p[0])).load()


def finite_tensors(value):
    if isinstance(value,np.ndarray):
        if not np.isfinite(value).all(): raise ValueError("Nonfinite checkpoint tensor")
        return 1
    if isinstance(value,dict): return sum(finite_tensors(v) for v in value.values())
    if isinstance(value,(tuple,list)): return sum(finite_tensors(v) for v in value)
    return 0


def verify(root, baseline, cell):
    root=Path(root); manifest=read(root/"artifact_manifest.json")["artifacts"]
    for name,record in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()): raise ValueError("Manifest path escapes root")
        if not p.is_file() or p.stat().st_size!=record["bytes"] or sha(p)!=record["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    out={"status":"verified","original_manifest_files_verified":len(manifest),"runs":{},
         "scope":"downloaded artifacts; no retraining, no new H5 inference","baseline":{}}
    results=read(root/"results.json"); selection=read(root/"selection.json"); histories=[]
    if cell:
        tree=ast.parse(Path(cell).read_text(encoding="utf-8"))
        definitions={v.id:n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
                     for v in n.targets if isinstance(v,ast.Name) and v.id in ("MODEL","WORKER")}
        for name,key in [("model.py","MODEL"),("worker.py","WORKER")]:
            ast.parse(definitions[key])
            if (root/"code"/name).read_text(encoding="utf-8")!=definitions[key]: raise ValueError("Cell differs from actual executed source")
        out["reproduction_cell_sha256"]=sha(cell)
    for name,result in results.items():
        run=root/"runs"/name; config=result["config"]; hist=read(run/"history.json")
        if len(hist)!=100 or [r["epoch"] for r in hist]!=list(range(1,101)): raise ValueError("Incomplete epoch history")
        histories.append(hist); ckinfo={}
        for kind in ("best","latest"):
            path=run/f"{kind}.pt"; ck=checkpoint(path)
            if sha(path)!=result["checkpoints"][kind]["sha256"] or ck["config"]!=config: raise ValueError("Weight config/hash mismatch")
            if ck["epoch"]!=(result["best_epoch"] if kind=="best" else 100): raise ValueError("Weight epoch mismatch")
            if ck["history"]!=hist[:ck["epoch"]]: raise ValueError("Weight history mismatch")
            if ck["best"]!=min(r["val_mse"] for r in hist[:ck["epoch"]]): raise ValueError("Incorrect best selection")
            steps=sorted({int(s["step"].item()) for s in ck["optimizer"]["state"].values()})
            ckinfo[kind]={"sha256":sha(path),"epoch":ck["epoch"],"best_mse":ck["best"],
                          "tensor_count":finite_tensors(ck),"model_parameters":sum(v.size for v in ck["model"].values()),
                          "optimizer_step_counts":steps,"optimizer_groups":ck["optimizer"]["param_groups"]}
            del ck
        for split in ("RR","FR"):
            block=result["metrics"][split]; rows=read(run/f"{split}_per_scene.json")
            if block["rows"]!=rows or block["count"]!=20 or [r["scene_index"] for r in rows]!=list(range(20)):
                raise ValueError("Missing or inconsistent test scenes")
            for key,value in block["mean"].items():
                if not np.isclose(value,np.mean([r[key] for r in rows]),rtol=1e-12,atol=1e-12): raise ValueError("Scene mean mismatch")
            for i in range(20):
                with np.load(run/"predictions"/split/f"scene_{i:03}.npz",allow_pickle=False) as f:
                    pred=f["prediction_dn"]
                    expected_shape=(4,256,256) if split=="RR" else (4,512,512)
                    if pred.shape!=expected_shape: raise ValueError("Prediction shape mismatch")
                    if not np.isfinite(pred).all() or int(f["scene_index"])!=i: raise ValueError("Prediction content mismatch")
        if sorted(i["scene_index"] for i in result["images"])!=selection["zero_based_scene_indices"]: raise ValueError("Wrong example selection")
        for i in result["images"]:
            if not (run/i["file"]).is_file(): raise ValueError("Missing panel")
        gate=np.stack([np.asarray(read(run/f"gate_epoch_{e:03}.json")["values"]) for e in range(1,101)])
        if gate.shape!=(100,2,7,3,4) or not np.isfinite(gate).all(): raise ValueError("Invalid gate diagnostics")
        init=read(run/"initialization.json")
        out["runs"][name]={"checkpoints":ckinfo,"initialization":init,"test_scenes":{"RR":20,"FR":20},
            "prediction_arrays_verified":40,"fixed_panels_verified":5,"gate_epochs":100,
            "train_plus_validation_minutes":sum(r["train_seconds"]+r["validation_seconds"] for r in hist)/60,
            "last_gate_stage_means":gate[-1,:,0].mean((1,2)).tolist(),
            "last_gate_fraction_low":gate[-1,:,2].mean((1,2)).tolist(),
            "last_gate_fraction_high":gate[-1,:,3].mean((1,2)).tolist(),
            "last_injected_energy":gate[-1,:,5].mean((1,2)).tolist()}
    if len({out["runs"][n]["initialization"]["common_sha256"] for n in results})!=1: raise ValueError("Unequal common initialization")
    if [r["sample_order_sha256"] for r in histories[0]]!=[r["sample_order_sha256"] for r in histories[1]]: raise ValueError("Unequal A0/A1 order")
    out["common_initialization_equal"]=out["A0_A1_epoch_order_equal"]=True
    if baseline:
        baseline=Path(baseline); name="github_QB_baseline_k6_s42"
        b=read(baseline/"results.json")[name]; bc=b["config"]; bh=read(baseline/"runs"/name/"history.json")
        source=read(baseline/"code/source.json")
        if source["files"]["SSA-MRN/src/ssamrn/metrics.py"]["sha256"]!=sha(root/"code/metrics.py"): raise ValueError("Different metric implementation")
        if sha(baseline/"code/kaggle_worker.py")!=bc["worker_sha256"]: raise ValueError("B0 source mismatch")
        mapping={"epochs":"epochs","seed":"seed","batch":"batch_size","micro":"micro_batch","lr":"lr","amp":"amp","deterministic":"controlled"}
        for result in results.values():
            a=result["config"]
            if any(a[k]!=bc[v] for k,v in mapping.items()): raise ValueError("B0 condition mismatch")
            if a["compile"] or a["adam_impl"]!="foreach" or a["order_mode"]!="legacy_cpu_precomputed": raise ValueError("B0 execution mismatch")
            for split in ("train","val","RR","FR"):
                if any(a["data"][split][k]!=bc["data"][split][k] for k in ("sha256","shapes")): raise ValueError("B0 dataset mismatch")
        env=read(root/"environment.json"); benv=read(baseline/"environment.json")
        if any(env[k]!=benv[k] for k in ("torch","cuda","python","gpus")): raise ValueError("B0 environment mismatch")
        if [r["sample_order_sha256"] for r in bh]!=[r["sample_order_sha256"] for r in histories[0]]: raise ValueError("B0 order mismatch")
        bck={}
        for kind in ("best","latest"):
            p=baseline/"runs"/name/f"{kind}.pt"; ck=checkpoint(p)
            if sha(p)!=b["checkpoints"][kind]["sha256"] or ck["config"]!=bc: raise ValueError("B0 weights mismatch")
            if ck["history"]!=bh[:ck["epoch"]]: raise ValueError("B0 weight history mismatch")
            if ck["best_mse"]!=min(r["val_mse"] for r in bh[:ck["epoch"]]): raise ValueError("B0 best mismatch")
            bck[kind]={"sha256":sha(p),"epoch":ck["epoch"],"tensor_count":finite_tensors(ck),
                       "optimizer_step_counts":sorted({int(s["step"].item()) for s in ck["optimizer"]["state"].values()}),
                       "optimizer_groups":ck["optimizer"]["param_groups"]}
        b0g=bck["latest"]["optimizer_groups"][0]
        for n in results:
            ag=out["runs"][n]["checkpoints"]["latest"]["optimizer_groups"][0]
            if any(ag[k]!=b0g[k] for k in ("lr","betas","eps","weight_decay","amsgrad","foreach")): raise ValueError("Adam hyperparameter mismatch")
        out["baseline"]={"status":"posthoc_artifact_conditions_verified","origin":"existing committed kaggle_band_gated_hf baseline",
            "original_A0_A1_execution_B0_audit":False,"same_data_environment_hyperparameters_order":True,
            "architecture_capacity_input_path_differ":True,"checkpoints":bck,"metrics":b["metrics"],
            "delta_from_B0":{n:{s:{k:r["metrics"][s]["mean"][k]-v for k,v in b["metrics"][s]["mean"].items() if k in r["metrics"][s]["mean"]}
                                for s in ("RR","FR")} for n,r in results.items()}}
        expected_steps=100*math.ceil(bc["data"]["train"]["shapes"]["gt"][0]/bc["batch_size"])
        out["realized_optimizer_updates"]={"nominal_batches":expected_steps,
            "B0":bck["latest"]["optimizer_step_counts"],
            **{n:out["runs"][n]["checkpoints"]["latest"]["optimizer_step_counts"] for n in results},
            "limitation":"AMP GradScaler skipped updates differ; matched nominal recipe, not equal realized updates or bitwise parity"}
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("root",type=Path); p.add_argument("--baseline",type=Path)
    p.add_argument("--cell",type=Path); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    result=verify(args.root,args.baseline,args.cell)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"files":result["original_manifest_files_verified"],
                      "runs":list(result["runs"]),"baseline":result["baseline"].get("status")},ensure_ascii=False))
