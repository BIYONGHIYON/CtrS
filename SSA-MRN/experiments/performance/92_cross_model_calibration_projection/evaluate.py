import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common import benchmark_evaluate as evaluation
from common.baseline_models import Wrapped

MODELS = ("adwm", "arconv", "cannet", "fusionnet", "u2net")
CONDITIONS = ("original", "brightness", "projection", "both")
METRICS = ("PSNR", "SAM", "ERGAS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="WV2 RR H5 file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, choices=range(1, 21), default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.model == "cannet" and (device.type != "cuda" or torch.cuda.device_count() != 1):
        parser.error("CANNet needs exactly one visible CUDA GPU")

    sensor = "WV2"
    scale = evaluation.SENSORS[sensor][1]
    net, _ = evaluation.load_model(
        ROOT / "benchmarks" / args.model, "author", sensor, None,
        args.weights, str(device)
    )
    ops = Wrapped(net, sensor, project=True, calibrate_from="WV3").eval()
    ops.calibration.r_train = ops.calibration.r_train.to(device)
    rows = {condition: [] for condition in CONDITIONS}

    def project(output, ms):
        with torch.enable_grad():
            result, _ = ops.projector(output.detach().double(), ms.detach().double())
        return result.detach().to(output.dtype)

    for index, scene in evaluation.scenes(args.data, ("pan", "lms", "ms", "gt"), args.limit):
        pan, lms, ms = [
            torch.from_numpy((scene[key] / scale).astype("float32"))[None]
            for key in ("pan", "lms", "ms")
        ]
        inputs = tuple(tensor.to(device) for tensor in (pan, lms, ms))
        with torch.no_grad():
            original = net(*inputs).cpu()
            brightness = ops.calibration(*inputs).cpu()

        outputs = {
            "original": original,
            "brightness": brightness,
            "projection": project(original, ms),
            "both": project(brightness, ms),
        }
        for name, output in outputs.items():
            if output.shape != lms.shape or not torch.isfinite(output).all():
                raise RuntimeError(f"Invalid {name} output at scene {index}")
            prediction = output[0].double().numpy() * scale
            rows[name].append(dict(scene=index, **evaluation.rr_metrics(scene["gt"], prediction)))
        print(f"Scene {index + 1}/{args.limit} done", flush=True)

    if any(len(scores) != args.limit for scores in rows.values()):
        raise RuntimeError(f"Expected {args.limit} WV2 scenes per condition")

    means = {name: evaluation.mean(scores, METRICS) for name, scores in rows.items()}
    output = ROOT / "runtime" / f"{args.model}_wv2_ablation_{args.limit}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mean": means, "per_scene": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(means, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
