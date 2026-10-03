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
    links=''.join(f'<li><a href="{r["file"]}">예시 {i}: {r["scene"]} · 204밴드</a></li>' for i,r in enumerate(records,1))
    (args.output_dir/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>SSA-MRN 밴드 뷰어</title><h1>테스트 5장면 · 204밴드</h1><p>장면을 선택하면 204밴드 슬라이더가 열립니다. 각 HTML은 내려받아 오프라인에서도 사용할 수 있습니다.</p><ul>'+links+'</ul></html>',encoding='utf-8')

if __name__=='__main__': main()
