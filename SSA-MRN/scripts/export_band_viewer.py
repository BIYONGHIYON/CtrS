"""Export five self-contained 204-band HTML viewers from a fixed checkpoint."""
import argparse, base64, hashlib, io, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def render(sample, prediction, metadata):
    gt = sample['gt'].numpy()
    lr = sample['lr_hsi'].numpy()
    pred = prediction.numpy()
    if gt.shape[0] != 204 or pred.shape != gt.shape:
        raise ValueError('Expected matching 204-band cubes')
    mask = sample.get('valid_mask', torch.ones(gt.shape[1:], dtype=torch.bool)).numpy().astype(bool)
    limits = np.percentile(gt[:, mask], [1, 99], axis=1)
    bands = []
    for b in range(204):
        low, high = limits[:, b]
        def display(a):
            return png(np.rint(np.clip((a-low)/max(high-low, 1e-8), 0, 1)*255).astype('uint8'))
        bands.append([display(lr[b]), display(pred[b]), display(gt[b])])
    rgb = png(np.rint(sample['rgb'].permute(1,2,0).numpy().clip(0,1)*255).astype('uint8'))
    payload = json.dumps(dict(metadata, bands=bands, rgb=rgb, limits=limits.T.tolist(), mask=png(mask.astype('uint8')*255)), ensure_ascii=False).replace('</', '<\\/')
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SSA-MRN 204밴드 비교</title>
<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 16px;background:#101820;color:#eef}input[type=range]{width:65%}.panels{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}img{width:100%;image-rendering:pixelated}figure{margin:0}figcaption{margin:8px 0}button,input{font:inherit}p{line-height:1.6}details img{max-width:256px}@media(max-width:700px){.panels{grid-template-columns:repeat(2,1fr)}}</style>
<h1>SSA-MRN · 204밴드 비교</h1><p id="scene"></p><label>밴드 <input id="slider" type="range" min="1" max="204" value="70"> <input id="number" type="number" min="1" max="204" value="70" style="width:70px"></label><p id="range"></p>
<div class="panels"><figure><figcaption>LR HSI · 64×64</figcaption><img id="lr"></figure><figure><figcaption>RGB guide · 고정 컬러</figcaption><img id="rgb"></figure><figure><figcaption>예측 HSI · 256×256</figcaption><img id="pred"></figure><figure><figcaption>정답 HSI · 256×256</figcaption><img id="gt"></figure></div>
<p>밴드 번호는 1부터 시작합니다. 단일 밴드는 회색조로 표시하며 RGB guide는 바뀌지 않습니다. 각 밴드의 정답 유효 영역 1–99% 범위를 LR·예측·정답에 공통 적용했습니다. 밴드마다 대비 범위가 달라 밴드 간 밝기를 직접 비교할 수 없습니다. 표시용 8비트 PNG이며 원본 반사율·평가 데이터가 아닙니다. LR은 최근접 확대 표시입니다. 파장 메타데이터는 확인되지 않아 파장을 표시하지 않습니다.</p>
<details><summary>정합 유효 마스크 · 흰색이 평가 영역</summary><img id="mask"></details><details><summary>가중치·평가 근거</summary><pre id="meta" style="white-space:pre-wrap"></pre></details>
<script>const data=PAYLOAD;const slider=document.getElementById('slider'),number=document.getElementById('number');function update(v){v=Math.max(1,Math.min(204,Math.round(Number(v)||1)));slider.value=number.value=v;const b=v-1;document.getElementById('lr').src=data.bands[b][0];document.getElementById('pred').src=data.bands[b][1];document.getElementById('gt').src=data.bands[b][2];document.getElementById('range').textContent=`밴드 ${v}/204 (0-based index ${b}) · 공통 표시 범위 ${data.limits[b][0].toFixed(6)} ~ ${data.limits[b][1].toFixed(6)}`;}slider.oninput=()=>update(slider.value);number.oninput=()=>update(number.value);document.getElementById('rgb').src=data.rgb;document.getElementById('mask').src=data.mask;document.getElementById('scene').textContent=`${data.scene} · tile ${data.tile} · test sample ${data.sample_index} · epoch ${data.epoch}`;const {bands,rgb,mask,limits,...meta}=data;document.getElementById('meta').textContent=JSON.stringify(meta,null,2);update(70);</script></html>'''.replace('PAYLOAD', payload)



def render_index(records):
    cards=''.join(f'<a class="card" href="{r["file"]}"><span class="badge">SCENE {i:02}</span><h2>{r["scene"]}</h2><p>LR HSI · RGB · 예측 · 정답</p><span class="open">204밴드 살펴보기 →</span></a>' for i,r in enumerate(records,1))
    return """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SSA-MRN 밴드 뷰어</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101820;color:#e7edf4;font:16px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1100px;margin:auto;padding:64px 24px}.eyebrow{color:#73d6ce;font-size:13px;font-weight:700;letter-spacing:.12em}h1{font-size:clamp(28px,5vw,44px);line-height:1.25;margin:12px 0 20px}p{color:#b6c5d3}.intro{max-width:700px}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:28px 0 36px}.stats span{background:#1b2936;border:1px solid #33475a;border-radius:24px;padding:6px 16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}.card{display:block;background:#192735;border:1px solid #35495b;border-radius:16px;padding:24px;color:inherit;text-decoration:none;transition:background .15s,transform .15s}.card:hover{background:#223647;transform:translateY(-3px);border-color:#73d6ce}.card:focus-visible{outline:3px solid #73d6ce;outline-offset:4px}.badge{font-size:12px;letter-spacing:.1em;color:#73d6ce;font-weight:700}h2{font-size:21px;margin:12px 0}.card p{font-size:14px}.open{color:#94e5dd;font-weight:600;font-size:14px}footer{border-top:1px solid #33475a;margin-top:40px;padding-top:20px;font-size:14px;color:#aabccc}footer a{color:#94e5dd}@media(max-width:600px){main{padding:32px 18px}.grid{grid-template-columns:1fr}}
</style></head><body><main><div class="eyebrow">CtrS / SSA-MRN</div><h1>테스트 5장면 · 204밴드</h1><p class="intro">장면을 선택하고 슬라이더를 움직여 분광 밴드별 복원 결과를 비교하세요. LR HSI와 RGB 입력, 예측 HSI, 정답 HSI를 함께 확인할 수 있습니다.</p><div class="stats"><span>5개 독립 장면</span><span>204개 분광 밴드</span><span>합성 ×4 초해상도</span></div><section class="grid" aria-label="테스트 장면">"""+cards+"""</section><footer>최근 완료 실험: RGB별 12특징 · K=4 · 23탭 · epoch 99<br>밴드별 공통 대비로 표시한 정성 비교입니다. 각 장면 HTML은 오프라인에서도 사용할 수 있습니다.<br><a href="https://github.com/BIYONGHIYON/CtrS/tree/codex/ssa-mrn-triple34-tiles/SSA-MRN">연구 문서와 평가 근거 ↗</a></footer></main></body></html>"""

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--sample-indices',type=int,nargs=5,default=[12,0,172,72,252])
    args=parser.parse_args()
    if args.output_dir.exists(): parser.error('Use a new output directory')
    sys.path.insert(0,str(args.repo_root/'SSA-MRN/src'))
    from ssamrn.data.lib_hsi import LIBHSI
    from ssamrn.models.rgb_grouped import build_rgb_hsi_model
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.expected_sha256 and digest!=args.expected_sha256: parser.error('Checkpoint hash mismatch')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=state['config'];torch.set_num_threads(2)
    manifest=args.repo_root/config['alignment_manifest'] if config.get('alignment_manifest') else None
    if manifest and config.get('alignment_sha256') and hashlib.sha256(manifest.read_bytes()).hexdigest()!=config['alignment_sha256']: parser.error('Alignment hash mismatch')
    data=LIBHSI(args.data_root,'test',config['patch_size'],alignment_manifest=manifest,eval_layout=config.get('eval_layout','tiles'),degradation=config.get('degradation','bicubic'))
    model=build_rgb_hsi_model(config).eval();model.load_state_dict(state['model'])
    args.output_dir.mkdir(parents=True)
    records=[]
    for i,index in enumerate(args.sample_indices,1):
        sample=data[index]
        with torch.inference_mode(): prediction=model(sample['rgb'][None],sample['lr_hsi'][None])[0].float()
        metadata=dict(scene=sample['scene'],sample_index=index,tile=index%data.per_scene,epoch=state['epoch'],checkpoint_sha256=digest,split='test',inference_device='cpu',display='GT valid-mask percentile 1-99 per band, shared across HSI panels',bands_count=204)
        path=args.output_dir/f'sample_{i:02}.html';path.write_text(render(sample,prediction,metadata),encoding='utf-8')
        records.append(dict(metadata,file=path.name,bytes=path.stat().st_size));print(json.dumps(records[-1]),flush=True)
    (args.output_dir/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (args.output_dir/'index.html').write_text(render_index(records),encoding='utf-8')

if __name__=='__main__': main()
