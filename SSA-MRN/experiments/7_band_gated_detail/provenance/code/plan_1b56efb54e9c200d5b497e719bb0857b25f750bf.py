# # Kaggle 실험: SSA-MRN 밴드별 게이트 고주파 주입
#
# CtrS QB 실험의 기준 조건(K=6, MSE, Adam 1e-4, 100 epochs)을 따라 세 모델을 같은 시드로 비교합니다.
#
# - baseline: SSA-MRN 기준 모델
# - high_frequency: 저장소의 기존 고정 PAN 고주파 잔차 경로
# - band_gated_hf: PAN 고주파를 MS 밴드와 위치별로 선택해 주입하는 새 후보
#
# 기존 고정 고주파 경로는 앞선 QB 단일 시드에서 기준선보다 RR PSNR 0.0621 dB, FR QNR 0.003928 낮았습니다. 새 후보는 기준 모델 출력의 밴드별 가로·세로 기울기와 PAN 고주파를 보고 별도 게이트를 학습합니다. 추가 잔차 경로는 0으로 초기화해 시작 시 기준 모델의 출력을 보존합니다.
#
# ## 시작하기
#
# 1. Kaggle Notebook Settings에서 GPU Accelerator를 선택합니다.
# 2. CtrS 코드와 PanCollection H5가 포함된 Dataset을 Input에 연결합니다.
# 3. 아래 셀을 순서대로 실행합니다.
# 4. 기본 SEEDS=[42]는 세 모델을 한 번씩 비교하는 실행 확인입니다. 최종 반복 검증은 SEEDS=[42, 43, 44]로 실행하세요.
# 5. 모델 선택은 validation MSE의 최솟값만 사용하고, RR/FR test는 학습 후 평가합니다.
# 6. 결과는 /kaggle/working/ctrS_band_gated_hf 에 저장되고 마지막에 ZIP으로 묶입니다. Notebook Version을 저장할 때 output 저장을 켜면 내려받을 수 있습니다.
#
# QB test 장면은 저장소의 이전 실험에서도 사용됐으므로 새 독립 test가 아니라 후속 비교입니다. FR QNR은 저장소 설명대로 MATLAB resize 일치성이 확인되지 않은 잠정 지표입니다.

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, subprocess, importlib.util
missing = [name for name in ["h5py", "scipy", "skimage", "matplotlib"]
           if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import csv, hashlib, json, math, random, shutil, time, urllib.request, zipfile
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

SENSOR = "QB"
K = 6
EPOCHS = 100
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
SEEDS = [42]  # 최종 반복 검증에서는 [42, 43, 44]
VARIANTS = ["baseline", "high_frequency", "band_gated_hf"]
PATH_OVERRIDE = {"repo_root": None, "train": None, "val": None, "rr_test": None, "fr_test": None}
OUTPUT_DIR = Path("/kaggle/working/ctrS_band_gated_hf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_ROOT = Path("/kaggle/input")
if not INPUT_ROOT.exists():
    raise FileNotFoundError("Kaggle 환경에서 실행해야 합니다: /kaggle/input이 없습니다.")

def find_unique(filename):
    hits = [p for p in INPUT_ROOT.rglob("*") if p.is_file() and p.name.lower() == filename.lower()]
    if len(hits) != 1:
        if not hits:
            raise FileNotFoundError(f"{filename}을 찾지 못했습니다. Dataset을 Input에 연결하세요.")
        raise RuntimeError(f"{filename}이 여러 개입니다. PATH_OVERRIDE에 사용할 경로를 지정하세요:\n"
                           + "\n".join(str(p) for p in hits))
    return hits[0]

def get_path(key, filename):
    return Path(PATH_OVERRIDE[key]) if PATH_OVERRIDE.get(key) else find_unique(filename)

if PATH_OVERRIDE["repo_root"]:
    root = Path(PATH_OVERRIDE["repo_root"])
    if (root / "SSA-MRN").is_dir():
        root = root / "SSA-MRN"
    source_file = root / "src/ssamrn/models/ssa_mrn.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"SSA-MRN 소스가 없습니다: {source_file}")
    SSA_ROOT, SOURCE_ROOT = root, root / "src"
else:
    hits = [p for p in INPUT_ROOT.rglob("ssa_mrn.py")
            if p.is_file() and "ssamrn" in p.parts and "models" in p.parts]
    if len(hits) != 1:
        raise RuntimeError("Kaggle Input에서 SSA-MRN/src/ssamrn/models/ssa_mrn.py를 하나 찾지 못했습니다. "
                           "PATH_OVERRIDE['repo_root']를 지정하세요.")
    source_file = hits[0]
    SOURCE_ROOT, SSA_ROOT = source_file.parents[2], source_file.parents[3]

TRAIN_H5 = get_path("train", "train_qb.h5").resolve()
VAL_H5 = get_path("val", "valid_qb.h5").resolve()
RR_TEST_H5 = get_path("rr_test", "test_qb_multiExm1.h5").resolve()
FR_TEST_H5 = get_path("fr_test", "test_qb_OrigScale_multiExm1.h5").resolve()
if TRAIN_H5 == VAL_H5:
    raise ValueError("Train과 validation 파일은 달라야 합니다.")

# 읽기 전용 Kaggle Input을 건드리지 않고 소스만 작업 디렉터리에 복사합니다.
RUNTIME_ROOT = Path("/kaggle/working/ctrS_runtime/SSA-MRN")
shutil.copytree(SOURCE_ROOT, RUNTIME_ROOT / "src", dirs_exist_ok=True)
upstream = SSA_ROOT / "references/upstream/network.py"
target = RUNTIME_ROOT / "references/upstream/network.py"
target.parent.mkdir(parents=True, exist_ok=True)
if upstream.is_file():
    shutil.copy2(upstream, target)
else:
    url = ("https://raw.githubusercontent.com/zhouchuanxu/SSA-MRN/"
           "a4ca40e407b12bf4c30f804384405ce321d11c51/network.py")
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:
        raise RuntimeError("공식 submodule network.py가 Input에 없습니다. "
                           "references/upstream/network.py를 Dataset에 넣거나 Kaggle Internet을 켜세요.") from exc

sys.path.insert(0, str(RUNTIME_ROOT / "src"))
from ssamrn.data.pancollection import PanCollectionH5, SENSOR_MAX, inspect_h5
from ssamrn.metrics import rr_metrics, fr_metrics
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle Settings에서 GPU Accelerator를 선택하세요.")
for label, path in [("train", TRAIN_H5), ("validation", VAL_H5),
                    ("RR test", RR_TEST_H5), ("FR test", FR_TEST_H5)]:
    print(label, path)
print("Train:", inspect_h5(TRAIN_H5))
print("Validation:", inspect_h5(VAL_H5))
print("RR test:", inspect_h5(RR_TEST_H5))
print("FR test:", inspect_h5(FR_TEST_H5))
train_check, val_check = PanCollectionH5(TRAIN_H5, SENSOR), PanCollectionH5(VAL_H5, SENSOR)
assert train_check.channels == val_check.channels
print("Train patches:", len(train_check), "| validation:", len(val_check), "| bands:", train_check.channels)
train_check.close()
val_check.close()

class FixedHighFrequency(RestoredPansharpeningNet):
    def __init__(self, channels, ssai_dimension):
        super().__init__(channels, ssai_dimension)
        with torch.random.fork_rng(devices=[]):
            self.detail = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(16, channels, 3, padding=1))
            nn.init.zeros_(self.detail[-1].weight)
            nn.init.zeros_(self.detail[-1].bias)

    @staticmethod
    def highpass(pan):
        w = pan.new_tensor([1, 4, 6, 4, 1]) / 16
        kernel = (w[:, None] * w[None, :]).view(1, 1, 5, 5)
        rows = torch.arange(-2, pan.shape[-2] + 2, device=pan.device).clamp(0, pan.shape[-2] - 1)
        cols = torch.arange(-2, pan.shape[-1] + 2, device=pan.device).clamp(0, pan.shape[-1] - 1)
        smooth = F.conv2d(pan.index_select(-2, rows).index_select(-1, cols), kernel)
        return pan - smooth

    def forward(self, pan, lms, ms):
        return super().forward(pan, lms, ms) + self.detail(self.highpass(pan))


class BandGatedHighFrequency(RestoredPansharpeningNet):
    def __init__(self, channels, ssai_dimension):
        super().__init__(channels, ssai_dimension)
        self.band_count = channels
        with torch.random.fork_rng(devices=[]):
            self.detail = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(16, channels, 3, padding=1))
            self.gate = nn.Sequential(nn.Conv2d(3 * channels, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, channels, 1))
            nn.init.zeros_(self.detail[-1].weight)
            nn.init.zeros_(self.detail[-1].bias)
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -2.0)

    @staticmethod
    def highpass(pan):
        w = pan.new_tensor([1, 4, 6, 4, 1]) / 16
        kernel = (w[:, None] * w[None, :]).view(1, 1, 5, 5)
        rows = torch.arange(-2, pan.shape[-2] + 2, device=pan.device).clamp(0, pan.shape[-2] - 1)
        cols = torch.arange(-2, pan.shape[-1] + 2, device=pan.device).clamp(0, pan.shape[-1] - 1)
        smooth = F.conv2d(pan.index_select(-2, rows).index_select(-1, cols), kernel)
        return pan - smooth

    def forward_components(self, pan, lms, ms):
        base = super().forward(pan, lms, ms)
        high = self.highpass(pan).expand(-1, self.band_count, -1, -1)
        detached = base.detach()
        dx = F.pad(detached[..., 1:] - detached[..., :-1], (0, 1, 0, 0))
        dy = F.pad(detached[..., 1:, :] - detached[..., :-1, :], (0, 0, 0, 1))
        gate = torch.sigmoid(self.gate(torch.cat([high, dx.abs(), dy.abs()], dim=1)))
        correction = gate * self.detail(self.highpass(pan))
        return base, gate, correction

    def forward(self, pan, lms, ms):
        base, _, correction = self.forward_components(pan, lms, ms)
        return base + correction


def build_model(variant, channels, k):
    if variant == "baseline":
        return RestoredPansharpeningNet(channels, k)
    if variant == "high_frequency":
        return FixedHighFrequency(channels, k)
    if variant == "band_gated_hf":
        return BandGatedHighFrequency(channels, k)
    raise ValueError(variant)

DEVICE = torch.device("cuda")
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value

def atomic_save(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)

def write_text(path, content):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


@torch.no_grad()
def validation_metrics(model, loader):
    model.eval()
    total = 0.0
    elements = 0
    psnr_sum = 0.0
    sam_sum = 0.0
    images = 0
    for batch in loader:
        values = {k: batch[k].to(DEVICE, non_blocking=True) for k in ("pan", "lms", "ms", "gt")}
        n = len(values["gt"])
        for start in range(0, n, MICRO_BATCH_SIZE):
            end = min(start + MICRO_BATCH_SIZE, n)
            pred = model(values["pan"][start:end], values["lms"][start:end], values["ms"][start:end])
            gt = values["gt"][start:end]
            diff = pred - gt
            total += diff.square().sum().item()
            elements += gt.numel()
            band_mse = diff.square().mean((-2, -1))
            psnr_sum += (-10 * torch.log10(band_mse.clamp_min(1e-30))).mean(1).sum().item()
            norms = pred.norm(dim=1) * gt.norm(dim=1)
            valid = norms > 0
            cosine = ((pred * gt).sum(1) / norms.clamp_min(1e-30)).clamp(-1, 1)
            angle = torch.rad2deg(torch.acos(cosine))
            counts = valid.flatten(1).sum(1)
            if (counts == 0).any():
                raise ValueError("Validation SAM has no valid pixels.")
            sam_sum += ((angle * valid).flatten(1).sum(1) / counts).sum().item()
            images += len(gt)
    return total / elements, psnr_sum / images, sam_sum / images

def train_config(variant, seed):
    return {"sensor": SENSOR, "variant": variant, "k": K, "seed": int(seed), "epochs": EPOCHS,
            "effective_batch": EFFECTIVE_BATCH_SIZE, "micro_batch": MICRO_BATCH_SIZE,
            "lr": LEARNING_RATE, "loss": "MSE", "train": str(TRAIN_H5), "val": str(VAL_H5),
            "precision": "FP32"}

def train_one(variant, seed):
    config = train_config(variant, seed)
    run_id = f"{SENSOR}_{variant}_k{K}_s{seed}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = run_dir / "latest.pt", run_dir / "best.pt"
    history_path, complete_path = run_dir / "history.jsonl", run_dir / "complete.json"
    if complete_path.exists():
        old = json.loads(complete_path.read_text(encoding="utf-8"))
        if old.get("config") == config and best_path.exists():
            print("Already complete; reuse", run_id)
            return run_id

    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = PanCollectionH5(TRAIN_H5, SENSOR), PanCollectionH5(VAL_H5, SENSOR)
    if train_set.channels != val_set.channels:
        raise ValueError("Train/validation band counts differ.")
    train_loader = DataLoader(train_set, batch_size=EFFECTIVE_BATCH_SIZE, shuffle=True,
                              generator=generator, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=EFFECTIVE_BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    model = build_model(variant, train_set.channels, K).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_epoch, best_mse = 0, float("inf")
    if latest_path.exists():
        ck = torch.load(latest_path, map_location="cpu", weights_only=True)
        if ck["config"] != config:
            raise ValueError(f"{run_id} 기존 checkpoint 설정과 현재 설정이 다릅니다.")
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["cpu_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        generator.set_state(ck["loader_rng"])
        start_epoch, best_mse = int(ck["epoch"]), float(ck["best_mse"])
        print("Resume", run_id, "at epoch", start_epoch)

    history = []
    if history_path.exists():
        history = [json.loads(x) for x in history_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        history = [x for x in history if x["epoch"] <= start_epoch]

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total, count, started = 0.0, 0, time.time()
        for batch in train_loader:
            values = {k: batch[k].to(DEVICE, non_blocking=True) for k in ("pan", "lms", "ms", "gt")}
            n = len(values["gt"])
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, n, MICRO_BATCH_SIZE):
                end = min(start + MICRO_BATCH_SIZE, n)
                pred = model(values["pan"][start:end], values["lms"][start:end], values["ms"][start:end])
                target = values["gt"][start:end]
                loss = F.mse_loss(pred, target)
                (loss * ((end - start) / n)).backward()
                total += F.mse_loss(pred.detach(), target, reduction="sum").item()
                count += target.numel()
            optimizer.step()

        train_mse = total / count
        val_mse, val_psnr, val_sam = validation_metrics(model, val_loader)
        if not math.isfinite(val_mse):
            raise RuntimeError("Validation MSE is non-finite.")
        improved = val_mse < best_mse
        best_mse = min(best_mse, val_mse)
        row = {"epoch": epoch + 1, "train_mse": train_mse, "val_mse": val_mse,
               "val_psnr_band_mean_peak1": val_psnr, "val_sam_deg": val_sam,
               "seconds": time.time() - started}
        history = [x for x in history if x["epoch"] < epoch + 1] + [row]
        write_text(history_path, "".join(json.dumps(x) + "\n" for x in history))
        ck = {"config": config, "epoch": epoch + 1, "best_mse": best_mse,
              "model": cpu_state(model.state_dict()), "optimizer": cpu_state(optimizer.state_dict()),
              "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
              "loader_rng": generator.get_state()}
        atomic_save(ck, latest_path)
        if improved:
            atomic_save(ck, best_path)
        print(f"{run_id} {epoch + 1}/{EPOCHS} train={train_mse:.8g} "
              f"val={val_mse:.8g} best={best_mse:.8g} sec={row['seconds']:.1f}", flush=True)

    write_text(complete_path, json.dumps({"config": config, "epoch": EPOCHS, "best_mse": best_mse}, indent=2))
    train_set.close()
    val_set.close()
    del model, optimizer, train_loader, val_loader, train_set, val_set
    torch.cuda.empty_cache()
    return run_id

# ## 4. 학습 실행
#
# 중단된 세션에서 같은 설정으로 다시 실행하면 epoch별 latest.pt에서 이어갑니다. 세션 자체가 초기화되면 이전 Notebook Version의 output을 새 Notebook Input으로 연결해야 checkpoint를 가져올 수 있습니다.

run_ids = []
for seed in SEEDS:
    for variant in VARIANTS:
        run_ids.append(train_one(variant, seed))
print("Runs:", run_ids)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
metric_specs = [
    ("train_mse", "Train MSE"),
    ("val_mse", "Validation MSE"),
    ("val_psnr_band_mean_peak1", "Validation band-mean PSNR (peak=1)"),
    ("val_sam_deg", "Validation SAM (degrees)"),
]
for run_id in run_ids:
    history_file = OUTPUT_DIR / run_id / "history.jsonl"
    hist = [json.loads(x) for x in history_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    for ax, (key, title) in zip(axes, metric_specs):
        ax.plot([row["epoch"] for row in hist], [row[key] for row in hist], label=run_id)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
axes[0].set_yscale("log")
axes[1].set_yscale("log")
axes[-1].legend(fontsize=7)
fig.tight_layout()
curve_path = OUTPUT_DIR / "learning_curves.png"
fig.savefig(curve_path, dpi=160)
display(fig)
plt.close(fig)
print("Saved:", curve_path)

# ## 5. RR·FR 전체 테스트
#
# 각 run의 best validation MSE 체크포인트만 평가합니다. RR은 GT 기반 지표, FR은 정답이 없는 QNR 계열 지표로 나눠 저장합니다. test 결과는 가중치 선택에 사용하지 않습니다.

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


from PIL import Image, ImageDraw

def save_rr_panel(arrays, prediction_dn, output_path, index):
    bands = [2, 1, 0]
    reference = arrays["gt"][bands]
    height, width = reference.shape[-2:]
    low = np.percentile(reference, 1, axis=(1, 2))
    high = np.percentile(reference, 99, axis=(1, 2))

    def rgb(chw):
        resized = []
        for band in chw[bands]:
            image = Image.fromarray(band.astype(np.float32), mode="F")
            image = image.resize((width, height), Image.Resampling.NEAREST)
            resized.append(np.asarray(image, dtype=np.float32))
        values = np.stack(resized, axis=-1)
        values = np.clip((values - low) / np.maximum(high - low, 1e-8), 0, 1)
        return Image.fromarray((values * 255).astype(np.uint8), mode="RGB")

    pan = arrays["pan"][0]
    p0, p1 = np.percentile(pan, [1, 99])
    pan_image = Image.fromarray(
        (np.clip((pan - p0) / max(float(p1 - p0), 1e-8), 0, 1) * 255).astype(np.uint8)
    ).convert("RGB")
    panels = [
        ("LR MS", rgb(arrays["ms"])),
        ("PAN", pan_image),
        ("Prediction", rgb(prediction_dn)),
        ("GT", rgb(arrays["gt"])),
    ]
    canvas = Image.new("RGB", (4 * width, height + 38), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), f"QB RR index {index} | full scene | shared GT 1-99 percentile stretch", fill="black")
    for j, (label, image) in enumerate(panels):
        canvas.paste(image, (j * width, 30))
        draw.text((j * width + 8, 17), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

with h5py.File(RR_TEST_H5, "r") as rr_file:
    rr_count = len(rr_file["pan"])
if rr_count < 5:
    raise ValueError("RR test must contain at least five distinct scenes.")
scene_indices = sorted(np.random.default_rng(42).choice(rr_count, 5, replace=False).tolist())
selection = {
    "seed": 42,
    "protocol": "RR",
    "zero_based_scene_indices": scene_indices,
    "scene_id": "H5 row index; source acquisition IDs are unavailable",
    "tile": "full scene",
    "order": "ascending H5 row index",
    "rule": "fixed before test evaluation; QB test was used in earlier CtrS experiments",
}
(OUTPUT_DIR / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

source_files = [
    RUNTIME_ROOT / "src/ssamrn/models/ssa_mrn.py",
    RUNTIME_ROOT / "src/ssamrn/data/pancollection.py",
    RUNTIME_ROOT / "src/ssamrn/metrics.py",
    RUNTIME_ROOT / "references/upstream/network.py",
]
source_hashes = {p.relative_to(RUNTIME_ROOT).as_posix(): sha256_file(p) for p in source_files}

def evaluate_run(run_id, protocol, path):
    ck_path = OUTPUT_DIR / run_id / "best.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    channels = 4 if cfg["sensor"] in ("QB", "GF2") else 8
    model = build_model(cfg["variant"], channels, cfg["k"]).to(DEVICE)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    scale = SENSOR_MAX[cfg["sensor"]]
    required = {"pan", "lms", "ms"} | ({"gt"} if protocol == "RR" else set())
    rows = []
    with h5py.File(path, "r") as h5:
        missing = required - set(h5.keys())
        if missing:
            raise ValueError(f"{protocol} H5 missing arrays: {sorted(missing)}")
        for index in range(len(h5["pan"])):
            arrays = {key: np.asarray(h5[key][index], dtype=np.float32) for key in required}
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError(f"Non-finite values: {protocol} sample {index}")
            inputs = [torch.from_numpy(arrays[k] / scale).unsqueeze(0).to(DEVICE)
                      for k in ("pan", "lms", "ms")]
            with torch.inference_mode():
                pred_norm = model(*inputs)[0].cpu().numpy().astype(np.float64)
            if not np.isfinite(pred_norm).all():
                raise ValueError(f"Non-finite prediction: {protocol} sample {index}")
            pred_dn = pred_norm * scale
            if protocol == "RR":
                metrics = rr_metrics(arrays["gt"].astype(np.float64), pred_dn)
                metrics["MSE_peak1"] = float(np.mean((pred_norm - arrays["gt"] / scale) ** 2))
            else:
                metrics = fr_metrics(pred_dn, arrays["lms"], arrays["ms"], arrays["pan"])
            rows.append({"index": index, **{k: float(v) for k, v in metrics.items()}})
            if protocol == "RR" and cfg["variant"] != "baseline" and index in scene_indices:
                image_path = OUTPUT_DIR / "images" / f"{run_id}_scene_{index:02d}.png"
                save_rr_panel(arrays, pred_dn, image_path, index)
            if (index + 1) % 5 == 0 or index + 1 == len(h5["pan"]):
                print(f"{run_id} {protocol}: {index + 1}/{len(h5['pan'])}")
    metric_keys = [key for key in rows[0] if key != "index"]
    mean = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    result = {"file": str(path), "sha256": sha256_file(path), "count": len(rows),
              "best_epoch": int(ck["epoch"]), "mean": mean, "per_sample": rows}
    del model, ck
    torch.cuda.empty_cache()
    return result

report = {
    "experiment": "QB_K6_band_gated_high_frequency",
    "selection": "best validation MSE only; test is not used to choose weights",
    "training": {"sensor": SENSOR, "k": K, "epochs": EPOCHS, "batch": EFFECTIVE_BATCH_SIZE,
                 "micro_batch": MICRO_BATCH_SIZE, "lr": LEARNING_RATE, "loss": "MSE",
                 "seeds": SEEDS, "variants": VARIANTS},
    "limitations": [
        "QB test scenes were used by earlier CtrS experiments; this is a follow-up comparison, not an independent test.",
        "FR D_s and QNR depend on scikit-image PAN resize; MATLAB parity remains unverified.",
        "One seed is only a pilot; repeat paired variants across seeds before broad improvement claims."
    ],
    "runs": {}
}
for run_id in run_ids:
    seed = int(run_id.rsplit("_s", 1)[1])
    variant = run_id.split("_k", 1)[0].replace(SENSOR + "_", "")
    history = [json.loads(x) for x in
               (OUTPUT_DIR / run_id / "history.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    report["runs"][run_id] = {
        "seed": seed, "variant": variant, "history": history,
        "RR": evaluate_run(run_id, "RR", RR_TEST_H5),
        "FR": evaluate_run(run_id, "FR", FR_TEST_H5),
    }
    (OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

paired = {}
for seed in SEEDS:
    baseline_id = f"{SENSOR}_baseline_k{K}_s{seed}"
    paired[str(seed)] = {}
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        run_id = f"{SENSOR}_{variant}_k{K}_s{seed}"
        paired[str(seed)][variant] = {}
        for protocol in ("RR", "FR"):
            base = report["runs"][baseline_id][protocol]["mean"]
            candidate = report["runs"][run_id][protocol]["mean"]
            paired[str(seed)][variant][protocol] = {
                key: float(candidate[key] - base[key]) for key in candidate
            }
report["paired_deltas_candidate_minus_baseline"] = paired

if len(SEEDS) > 1:
    summary = {}
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        summary[variant] = {}
        for protocol in ("RR", "FR"):
            keys = paired[str(SEEDS[0])][variant][protocol]
            summary[variant][protocol] = {
                key: {
                    "mean": float(np.mean([paired[str(seed)][variant][protocol][key] for seed in SEEDS])),
                    "sample_std": float(np.std(
                        [paired[str(seed)][variant][protocol][key] for seed in SEEDS], ddof=1)),
                    "values": [paired[str(seed)][variant][protocol][key] for seed in SEEDS],
                }
                for key in keys
            }
    report["paired_delta_seed_summary"] = summary

(OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
csv_rows = []
for run_id, result in report["runs"].items():
    for protocol in ("RR", "FR"):
        for row in result[protocol]["per_sample"]:
            csv_rows.append({"run_id": run_id, "seed": result["seed"], "variant": result["variant"],
                             "protocol": protocol, **row})
metric_names = sorted({k for row in csv_rows for k in row if k not in
                       {"run_id", "seed", "variant", "protocol", "index"}})
with (OUTPUT_DIR / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["run_id", "seed", "variant", "protocol", "index", *metric_names])
    writer.writeheader()
    writer.writerows(csv_rows)
print(json.dumps(paired, indent=2))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for run_id, result in report["runs"].items():
    hist = result["history"]
    axes[0].plot([r["epoch"] for r in hist], [r["train_mse"] for r in hist], label=run_id)
    axes[1].plot([r["epoch"] for r in hist], [r["val_mse"] for r in hist], label=run_id)
axes[0].set_title("Train MSE")
axes[1].set_title("Validation MSE")
axes[0].set_yscale("log")
axes[1].set_yscale("log")
for ax in axes[:2]:
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)

labels, psnr, sam, qnr = [], [], [], []
for seed in SEEDS:
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        delta = paired[str(seed)][variant]
        labels.append(f"s{seed} {variant}")
        psnr.append(delta["RR"]["PSNR"])
        sam.append(-delta["RR"]["SAM"])
        qnr.append(delta["FR"]["QNR"])
x = np.arange(len(labels))
axes[2].bar(x - 0.25, psnr, width=0.25, label="RR PSNR delta (higher is better)")
axes[2].bar(x, sam, width=0.25, label="negative RR SAM delta")
axes[2].bar(x + 0.25, qnr, width=0.25, label="FR QNR delta (provisional)")
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
axes[2].set_title("Candidate minus same-seed baseline")
axes[2].grid(axis="y", alpha=0.25)
axes[2].legend(fontsize=7)
axes[0].legend(fontsize=7)
fig.tight_layout()
plot_path = OUTPUT_DIR / "training_and_test_deltas.png"
fig.savefig(plot_path, dpi=160)
display(fig)
plt.close(fig)

# 보존 자료의 크기와 SHA-256을 남깁니다.
manifest = {}
for artifact in sorted(OUTPUT_DIR.rglob("*")):
    if artifact.is_file() and artifact.name != "artifact_manifest.json":
        manifest[str(artifact.relative_to(OUTPUT_DIR))] = {
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
(OUTPUT_DIR / "artifact_manifest.json").write_text(
    json.dumps({"artifacts": manifest, "source_sha256": source_hashes}, indent=2),
    encoding="utf-8",
)

zip_path = Path("/kaggle/working/ctrS_band_gated_hf_results.zip")
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT_DIR.parent))
print("ZIP:", zip_path, "| MiB:", round(zip_path.stat().st_size / (1024 ** 2), 1))
print("Output folder:", OUTPUT_DIR)

# ## 산출물 / 결과 해석
#
# - 각 run 폴더에는 best.pt, latest.pt, history.jsonl, complete.json이 있습니다.
# - test_metrics.json에는 RR/FR 전체 장면 결과, 장면별 값, 파일 SHA-256, paired 차이가 저장됩니다.
# - per_sample_metrics.csv는 장면별 RR/FR 표입니다.
# - training_and_test_deltas.png는 학습 곡선과 baseline 대비 test 차이입니다.
# - ctrS_band_gated_hf_results.zip을 Kaggle Output에서 내려받습니다.
#
# PSNR은 높을수록, SAM은 낮을수록 좋습니다. 게이트 모델은 baseline과 기존 고정 high_frequency 경로 모두와 비교하세요. 새 후보의 이득이 여러 시드에서 반복되지 않거나 RR과 FR 지표가 엇갈리면 SSA-MRN의 보편적 개선으로 결론내리지 않습니다.
