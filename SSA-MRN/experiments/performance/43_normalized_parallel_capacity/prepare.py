"""Verify immutable completed artifacts and build the Korean experiment report."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,ast,json,hashlib,math,statistics,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
ARMS=('C0','C1','C2','C3');SEEDS=(42,43,44)
DIRECTION={'PSNR':1,'SAM':-1,'ERGAS':-1,'SCC':1,'Q2n':1,'D_lambda':-1,'D_s':-1,'QNR':1}
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def job(a,s):return f'{a}_QB_s{s}'
def fingerprint(cfg):return hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
def immutable_check():
 m=read(HERE/'provenance/server_export_manifest.json')
 for name,e in m['files'].items():
  p=HERE/name
  assert p.is_file() and p.stat().st_size==e['bytes'] and digest(p)==e['sha256'],name
 cfg=read(HERE/'provenance/runtime_config.json');assert fingerprint(cfg)==cfg['fingerprint']
 for name,value in cfg['source_hashes'].items():assert digest(HERE/name)==value,name
 assert read(HERE/'provenance/all_complete.json')['training_runs']==12
 for label in ['initial_training','resumed_training']:
  contract=read(HERE/('provenance/initial_runtime_config.json' if label=='initial_training' else 'provenance/runtime_config.json'))
  for name,h in contract['source_hashes'].items():assert digest(HERE/'provenance/source_snapshots'/label/name)==h,(label,name)
 return dict(raw_files=len(m['files']),raw_checkpoints=sum(n.endswith('.pt') for n in m['files']),original_server_export_hashes=True,current_and_initial_training_source_hashes=True),cfg

def collect(cfg):
 runs={};orders={};verification=[]
 for a in ARMS:
  for seed in SEEDS:
   key=job(a,seed);r=HERE/'results'/key;c=HERE/'checkpoints'/key
   h=read(r/'history.json');complete=read(r/'training_complete.json');ev=read(r/'evaluation_complete.json')
   assert [x['epoch'] for x in h]==list(range(151)),key
   best=min(h,key=lambda x:(x['val_mse'],x['epoch']));assert complete['best_epoch']==best['epoch']==ev['best_epoch']
   assert complete['best_val']['val_mse']==best['val_mse'];assert complete['epoch']==150 and complete['attempted']==80400
   assert complete['best_sha256']==digest(c/'best.pt')==ev['weight_sha256'];assert complete['latest_sha256']==digest(c/'latest.pt')
   for epoch,row in enumerate(h[1:],1):
    expected=cfg['lr_end']+.5*(cfg['lr_start']-cfg['lr_end'])*(1+math.cos(math.pi*(epoch-1)/(cfg['epochs']-1)))
    assert abs(row['lr']-expected)<1e-15
    assert row['attempted']==epoch*536 and row['updates']+row['skipped']==row['attempted']
    assert all(math.isfinite(row[k]) for k in ('train_mse','val_mse','val_psnr','val_sam','epoch_seconds'))
    k=(seed,epoch)
    if k in orders:assert orders[k]==row['order_sha256']
    orders[k]=row['order_sha256']
   assert h[-1]['updates']==complete['updates']
   protos={}
   for proto in ('RR','FR'):
    data=read(r/'evaluation'/f'{proto}.json');assert [x['index'] for x in data['rows']]==list(range(20))
    assert data['weight_sha256']==ev['weight_sha256'] and data['best_epoch']==best['epoch']
    for metric,mean in data['mean'].items():
     values=[row['metrics'][metric] for row in data['rows']];assert all(math.isfinite(v) for v in values)
     assert abs(float(np.mean(values))-mean)<1e-12;assert ev['means'][proto][metric]==mean
    protos[proto]=data
   for index in cfg['scenes']:assert (r/'evaluation'/f'RR_scene_{index:03}.png').is_file()
   runs[key]=dict(arm=a,seed=seed,best_epoch=best['epoch'],val_mse=best['val_mse'],updates=complete['updates'],skipped=complete['attempted']-complete['updates'],history=h,protocols=protos)
   verification.append(dict(job=key,best_epoch=best['epoch'],minimum_validation_confirmed=True,history_epochs=151,attempted=complete['attempted'],updates=complete['updates'],skipped=complete['attempted']-complete['updates'],best_sha256=complete['best_sha256'],latest_sha256=complete['latest_sha256'],RR_scenes=20,FR_scenes=20))
 summary={'arms':{},'contrasts':{},'runs':verification,'limits':['원본 기준 유지','QB 기존 탐색 benchmark; 독립 holdout 아님','QNR/Q2n/SCC MATLAB 수치 대응 미검증','동일 장면을3seed 반복;60개 독립 장면 아님','FP16 AMP skip으로 실제 optimizer 업데이트 수가 군별로 다름']}
 for a in ARMS:
  rr={}
  for proto in ('RR','FR'):
   rr[proto]={}
   for metric in runs[job(a,42)]['protocols'][proto]['mean']:
    vals=[runs[job(a,s)]['protocols'][proto]['mean'][metric] for s in SEEDS]
    rr[proto][metric]=dict(mean=statistics.mean(vals),sample_sd=statistics.stdev(vals),by_seed={str(s):v for s,v in zip(SEEDS,vals)})
  summary['arms'][a]=rr
 comparisons={}
 for candidate,reference in [('C1','C0'),('C2','C0'),('C3','C0'),('C3','C1'),('C3','C2')]:
  label=candidate+'_vs_'+reference;comparisons[label]={};summary['contrasts'][label]={}
  for proto in ('RR','FR'):
   comparisons[label][proto]={};summary['contrasts'][label][proto]={}
   for metric in summary['arms'][candidate][proto]:
    deltas=[];seed_values=[]
    for seed in SEEDS:
     c=runs[job(candidate,seed)]['protocols'][proto];r=runs[job(reference,seed)]['protocols'][proto]
     sd=c['mean'][metric]-r['mean'][metric];seed_values.append(sd)
     for x,y in zip(c['rows'],r['rows']):deltas.append(dict(seed=seed,index=x['index'],delta=x['metrics'][metric]-y['metrics'][metric]))
    item=dict(mean_delta=statistics.mean(seed_values),sample_sd=statistics.stdev(seed_values),seed_deltas={str(s):v for s,v in zip(SEEDS,seed_values)},improved_seeds=sum(v*DIRECTION[metric]>0 for v in seed_values),improved_scene_seed_pairs=sum(x['delta']*DIRECTION[metric]>0 for x in deltas),total_scene_seed_pairs=60)
    summary['contrasts'][label][proto][metric]=item;comparisons[label][proto][metric]=dict(**item,pairs=deltas)
 summary['factorial_interaction']={proto:{k:summary['contrasts']['C3_vs_C2'][proto][k]['mean_delta']-summary['contrasts']['C1_vs_C0'][proto][k]['mean_delta'] for k in summary['arms']['C0'][proto]} for proto in ('RR','FR')}
 summary['classification']='부분 개선'
 # Classification uses the observed RR/FR directions, rather than validation to promote a model.
 stable=all(summary['contrasts']['C1_vs_C0'][proto][k]['improved_seeds']==3 for proto,k in [('RR','PSNR'),('RR','SAM'),('RR','ERGAS'),('FR','QNR'),('FR','D_lambda'),('FR','D_s')])
 summary['C1_all_major_metrics_all_seeds_improved']=stable
 summary['classification']='부분 개선'
 write(HERE/'results/summary.json',summary);write(HERE/'results/paired_scene_deltas.json',comparisons)
 return runs,summary

def check_weights(runs):
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from model import Model
 for seed in SEEDS:
  generator=torch.Generator().manual_seed(seed+100000)
  for epoch in range(1,151):
   order=torch.randperm(17139,generator=generator)
   assert hashlib.sha256(order.numpy().tobytes()).hexdigest()==runs[job('C0',seed)]['history'][epoch]['order_sha256']
 torch.manual_seed(730);inputs=[torch.rand(s) for s in [(1,1,64,64),(1,4,64,64),(1,4,16,16)]]
 records=[];initial={}
 paths=sorted((HERE/'checkpoints').glob('*/*.pt'))+sorted((HERE/'provenance/epoch1_checkpoints').glob('*/*.pt'))
 assert len(paths)==40
 for p in paths:
  ck=torch.load(p,map_location='cpu',weights_only=False);arm=ck['arm'];seed=ck['seed'];m=Model(arm,seed).eval();m.load_state_dict(ck['model'],strict=True)
  assert all(torch.isfinite(v).all() for v in ck['model'].values() if v.is_floating_point())
  with torch.inference_mode():y=m(*inputs)
  assert torch.isfinite(y).all()
  if p.name=='initial.pt':initial[(arm,seed)]={k:v.clone() for k,v in ck['model'].items()}
  if 'epoch1_checkpoints' not in p.parts:
   r=runs[job(arm,seed)]
   if p.name=='best.pt':assert ck['best_epoch']==r['best_epoch'] and ck['metrics']['val_mse']==r['val_mse']
   if p.name=='latest.pt':
    assert ck['epoch']==150 and ck['next_batch']==0 and ck['order'] is None and len(ck['history'])==151
    assert ck['updates']==r['updates'] and ck['attempted']==80400
    best=torch.load(p.with_name('best.pt'),map_location='cpu',weights_only=False)
    assert ck['best_epoch']==best['best_epoch']
    assert all(torch.equal(v,best['model'][k]) for k,v in ck['best_model'].items())
    assert all(int(st['step'])==r['updates'] for st in ck['optimizer']['state'].values())
  records.append(dict(path=str(p.relative_to(HERE)),sha256=digest(p),strict_load=True,cpu_fp32_finite_forward=True,parameters=sum(x.numel() for x in m.parameters())))
  del ck,m
 pairs=[]
 for seed in SEEDS:
  for a,b in [('C0','C1'),('C2','C3')]:assert all(torch.equal(v,initial[(b,seed)][k]) for k,v in initial[(a,seed)].items())
  outputs={}
  for a in ARMS:
   m=Model(a,seed).eval();m.load_state_dict(initial[(a,seed)],strict=True)
   with torch.inference_mode():outputs[a]=m(*inputs)
  for a,b in [('C0','C2'),('C1','C3')]:torch.testing.assert_close(outputs[a],outputs[b],rtol=0,atol=0)
  pairs.append(dict(seed=seed,initial_state_pairs_equal=[['C0','C1'],['C2','C3']],initial_output_pairs_exact=[['C0','C2'],['C1','C3']]))
 return dict(checkpoints=records,count=len(records),regenerated_full_epoch_permutations=450,initial_pair_checks=pairs,environment=dict(python=sys.version,torch=torch.__version__,cpu_threads=1,input_shapes=[list(x.shape) for x in inputs]),scope='원본40개 strict load·CPU FP32 합성 유한 forward;12개 latest optimizer step·best 상태 일치; H5 전체 재평가 아님')

def figures(runs,summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib import ticker
 from matplotlib import font_manager
 candidates=[Path('/System/Library/Fonts/AppleSDGothicNeo.ttc'),Path('C:/Windows/Fonts/malgun.ttf')]
 for f in candidates:
  if f.exists():font_manager.fontManager.addfont(str(f));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(f)).get_name();break
 plt.rcParams['axes.unicode_minus']=False;plt.rcParams['figure.dpi']=130
 colors={'C0':'#677585','C1':'#1b9e77','C2':'#d59116','C3':'#8664bd'};dest=HERE/'results/figures';dest.mkdir(parents=True,exist_ok=True)
 for metric,title in [('train_mse','학습 MSE'),('val_mse','Validation MSE'),('val_psnr','Validation PSNR'),('val_sam','Validation SAM')]:
  fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
  for ax,seed in zip(axes,SEEDS):
   for a in ARMS:
    h=runs[job(a,seed)]['history'];h=h[1:] if metric=='train_mse' else h
    ax.plot([x['epoch'] for x in h],[x[metric] for x in h],label=a,color=colors[a],lw=1.3)
   ax.set(title=f'seed{seed}',xlabel='에폭',ylabel=title);ax.grid(alpha=.2)
   if metric.endswith('mse'):
    ax.set_yscale('log');ax.yaxis.set_major_locator(ticker.LogLocator(base=10,subs=(1,2,5)));ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,_:f'{value:.1e}'));ax.yaxis.set_minor_formatter(ticker.NullFormatter())
  axes[0].legend(ncol=2);fig.suptitle(title+' · 전체150에폭 실제 기록');fig.savefig(dest/f'{metric}.png');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
 for ax,(proto,metric) in zip(axes.flat,[('RR','PSNR'),('RR','SAM'),('RR','ERGAS'),('FR','QNR'),('FR','D_lambda'),('FR','D_s')]):
  for i,a in enumerate(ARMS):
   x=summary['arms'][a][proto][metric];ax.bar(i,x['mean'],color=colors[a],alpha=.75,yerr=x['sample_sd'],capsize=3)
   ax.scatter([i]*3,list(x['by_seed'].values()),color='black',s=13,zorder=4)
  ax.set_xticks(range(4),ARMS);ax.set_title(proto+' '+metric+(' ↑' if DIRECTION[metric]>0 else ' ↓'));ax.grid(axis='y',alpha=.2)
  vals=[v for a in ARMS for v in summary['arms'][a][proto][metric]['by_seed'].values()];span=max(vals)-min(vals);ax.set_ylim(min(vals)-max(span*.25,1e-5),max(vals)+max(span*.3,1e-5))
 fig.suptitle('세 시드 평균±표본 SD · 점은 시드별20장 평균 · FR 지표는 잠정치');fig.savefig(dest/'comparison.png');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for ax,(proto,metric) in zip(axes,[('RR','PSNR'),('RR','SAM'),('FR','QNR')]):
  for a in ('C1','C2','C3'):
   items=[]
   for seed in SEEDS:
    c=runs[job(a,seed)]['protocols'][proto]['rows'];b=runs[job('C0',seed)]['protocols'][proto]['rows'];items.append([x['metrics'][metric]-y['metrics'][metric] for x,y in zip(c,b)])
   ax.plot(range(1,21),np.mean(items,axis=0),marker='.',label=a,color=colors[a],lw=1)
  ax.axhline(0,color='black',lw=.8);ax.set(title=proto+' Δ'+metric,xlabel='장면 순서(1기준)',ylabel='원본 대비 차이');ax.grid(alpha=.2)
 axes[0].legend();fig.suptitle('동일20장면의 차이 · 각 점은 같은 장면의3시드 평균');fig.savefig(dest/'scene_deltas.png');plt.close(fig)


def fmt(x,n=6):return f'{x:.{n}f}'
def report(runs,s,verify):
 cfg=read(HERE/'provenance/runtime_config.json');lines=[]
 add=lines.append
 add('# MS 기준 가역 정규화와 저해상도 병렬 특징 용량 확장\n')
 add('**연구 분야:** 기존 성능 개선\n\n**실험자:** 이병현\n')
 add('**변경 내용:** 원본 SSA-MRN에 입력별 MS 공통 가역 정규화(N1)와 저해상도 병렬 특징 가지(W2)를 단독·결합 적용한 네 군을 세 시드에서 비교했습니다.\n')
 add('| 확인된 결과 | 상태 | 개선 여부 |\n|---|:---:|:---:|')
 d=s['contrasts']['C3_vs_C0'];c=s['contrasts']['C3_vs_C2'];n=s['contrasts']['C1_vs_C0']
 add(f"| C3는 원본 대비 PSNR {d['RR']['PSNR']['mean_delta']:+.4f}dB이나 잠정 QNR {d['FR']['QNR']['mean_delta']:+.5f}였습니다. 정규화 단독 C1의 평균 RR·FR은 개선됐으나 결합의 FR 악화 완화는 실패했습니다. | 평가 완료·3seed | 부분 개선 |\n")
 add('## 1. 목적과 상태\n\n29번 W2의 RR 이득을 유지하면서 FR 공간 왜곡을 줄일 수 있는지 확인했습니다. 네 군×seed42·43·44의150에폭 학습과 각 validation best의 RR20·FR20 전체 평가를 마쳤습니다. 최종 학습은2026-10-19 17:43, 전체 평가는17:45(KST)에 끝났습니다. 같은20장면을12개 모델로 평가한 RR240·FR240회이며,480개 독립 영상 실험이 아닙니다. 기존 원본 기준 모델을 유지합니다.\n')
 add('## 2. 변경한 구조와 비교 조건\n\n| 군 | 입력·출력 변환 | Backbone | 파라미터 수 |\n|---|---|---|---:|\n| C0 | 없음 | 원본 | 426,654 |\n| C1 | N1 가역 정규화 | 원본 | 426,654 |\n| C2 | 없음 | 원본+W2 | 2,853,918 |\n| C3 | N1 가역 정규화 | 원본+W2 | 2,853,918 |\n')
 add('N1은 해당 LR MS의 밴드·공간 전체 모집단 평균 μ와 s=max(표준편차,0.01)를 구해 PAN·제공 LMS·MS에 같은(x−μ)/s를 적용하고 출력에 sy+μ를 적용합니다. GT·다른 샘플은 통계에 포함하지 않았습니다. W2는 backbone 입력 특징에 PixelUnshuffle(2) →256에서128채널1×1 Conv →128채널 잔차 블록8개 →128에서256채널1×1 Conv →PixelShuffle(2)를 병렬 적용해 원본 backbone 출력에 더합니다. 마지막 Conv는0으로 초기화했습니다. 실제 backbone과 추가 가지에는 BatchNorm이 없습니다.\n')
 add('아래는 C3의 흐름이며 다른 군의 차이는 위 표와 같습니다.\n')
 add('```mermaid\nflowchart LR\n    A[PAN · 제공 LMS · MS] --> N{C1·C3: MS 공통 정규화}\n    N --> C[원본 특징 추출]\n    C --> B[원본 backbone]\n    C --> W[C2·C3: 저해상도 W2 가지]\n    B --> S[특징 합산]\n    W --> S\n    S --> R[원본 SSAI·다중 해상도 복원]\n    R --> I[C1·C3: 역변환]\n    I --> O[최종 출력]\n```\n')
 add('| 조건 | 실제 수행 |\n|---|---|\n| 데이터 | QB train17,139·validation1,905·RR20·FR20, 원본 해시 확인 |\n| 모델·배율 | K4·공간4배, LRPAN·제공 LMS·원본 SSAI 유지 |\n| 초기화·순서 | 시드별 공통 core 초기값·150개 epoch 순서 해시 일치 |\n| 학습량 | 각150에폭, 전체 패치 사용·drop 없음, epoch당536회 업데이트 시도 |\n| 배치·optimizer | effective/micro32, Adam 기본 betas·eps·weight_decay0 |\n| 학습률 | 첫 에폭1e-4 →마지막 에폭1e-6, 에폭별 cosine |\n| 손실·선택 | DN/2047 좌표의 최종 출력 MSE; epoch0 포함 최소 validation MSE, 동률은 먼저 나온 에폭 |\n| 정밀도 | core AMP FP16; 통계·역변환·손실·validation·평가 FP32; TF32 off |\n| 실행 | RTX3060Ti8GB, 최대2개 병렬, CPU1스레드·공유 FP32 mmap·pinned 전송·channels-last |\n| CUDA 설정 | cuDNN autotune·deterministic convolution, deterministic algorithms warn-only·CUBLAS workspace 고정 |\n')
 add('실행 배치는 본 학습 전 두 W2 군의 처리 속도·메모리로 선택했습니다. validation/test로 최적화 조건을 고르지 않았습니다. 짧은 AMP 재개 검사를 통과했으며 첫 에폭 뒤 완료 로그 오류는 기존 모델·optimizer·scaler·RNG·shuffle·커서를 보존해 수정 후 재개했습니다. 원본 초기 학습 소스·설정·첫 에폭 가중치와 재개 근거는 provenance에 보존했습니다. FP16의 실제 업데이트 건너뜀은 아래에 모두 기록했으며 네 군의 실제 optimizer 업데이트 수가 완전히 같지는 않습니다.\n')
 add('## 3. 정량 결과\n\n### Validation 선택과 업데이트\n\n| 군 | seed | best epoch | best MSE | 실제 업데이트/80,400시도 | AMP skip |\n|---|---:|---:|---:|---:|---:|')
 for a in ARMS:
  for seed in SEEDS:
   r=runs[job(a,seed)];add(f"| {a} | {seed} | {r['best_epoch']} | {r['val_mse']:.12f} | {r['updates']:,} | {r['skipped']} |")
 add('\n### RR20·FR20의 세 시드 평균±표본 SD\n\n각 시드에서20장 평균을 구한 뒤 세 시드의 평균·표본 SD를 계산했습니다. 장면×시드60회를 독립 표본으로 취급하지 않았습니다.\n')
 for proto,metrics in [('RR',['PSNR','SAM','ERGAS','SCC','Q2n']),('FR',['QNR','D_lambda','D_s'])]:
  add('| 군 | '+' | '.join(k+(' ↑' if DIRECTION[k]>0 else ' ↓') for k in metrics)+' |\n|---|'+'---:|'*len(metrics))
  for a in ARMS:add('| '+a+' | '+' | '.join(f"{s['arms'][a][proto][k]['mean']:.6f} ± {s['arms'][a][proto][k]['sample_sd']:.6f}" for k in metrics)+' |')
  add('')
 add('### 원본 대비 시드별 차이\n\n| 군 | seed | ΔPSNR dB | ΔSAM ° | Δ잠정 QNR | ΔDλ | ΔDs |\n|---|---:|---:|---:|---:|---:|---:|')
 for a in ('C1','C2','C3'):
  for seed in SEEDS:
   delta=s['contrasts'][a+'_vs_C0'];get=lambda p,k:delta[p][k]['seed_deltas'][str(seed)]
   add(f"| {a} | {seed} | {get('RR','PSNR'):+.6f} | {get('RR','SAM'):+.6f} | {get('FR','QNR'):+.6f} | {get('FR','D_lambda'):+.6f} | {get('FR','D_s'):+.6f} |")
 add('\n### 장면별 개선 수\n\n| 군·대조군 | RR PSNR | RR SAM | FR QNR | FR Dλ | FR Ds |\n|---|---:|---:|---:|---:|---:|')
 for key in s['contrasts']:
  v=s['contrasts'][key];add('| '+key+' | '+' | '.join(str(v[p][k]['improved_scene_seed_pairs'])+'/60' for p,k in [('RR','PSNR'),('RR','SAM'),('FR','QNR'),('FR','D_lambda'),('FR','D_s')])+' |')
 add('\n각60회는 동일20장면의3시드 반복입니다. 정확한 시드·장면별 차이는[전체 paired 기록](results/paired_scene_deltas.json)에 보존했습니다.\n')
 add('## 4. 기준선 대비 차이와 해석\n')
 add(f"C1 정규화 단독은 원본 대비 세 시드 평균 PSNR {n['RR']['PSNR']['mean_delta']:+.6f}dB·SAM {n['RR']['SAM']['mean_delta']:+.6f}°·잠정 QNR {n['FR']['QNR']['mean_delta']:+.6f}였습니다. RR의 PSNR·SAM·ERGAS는3/3시드에서 개선됐지만 잠정 FR QNR은2/3시드에서만 개선됐습니다. seed43은 QNR −0.003282·Ds +0.001276·Dλ +0.002060으로 모두 악화했고, seed42도 Dλ는 +0.000631로 악화했습니다. 따라서 C1도 부분 개선이며 FR 이득의 시드 안정성을 확정하지 않습니다. 단일 센서·기존 탐색 benchmark 결과이며 기준 모델로 승격하지 않습니다.\n")
 add(f"C3 결합군은 원본 대비 PSNR {d['RR']['PSNR']['mean_delta']:+.6f}dB이나 잠정 QNR {d['FR']['QNR']['mean_delta']:+.6f}였습니다. C2 W2 단독 대비 PSNR {c['RR']['PSNR']['mean_delta']:+.6f}dB·SAM {c['RR']['SAM']['mean_delta']:+.6f}°·QNR {c['FR']['QNR']['mean_delta']:+.6f}·Ds {c['FR']['D_s']['mean_delta']:+.6f}로, 용량 확장의 FR 악화 완화 가설은 지지되지 않았습니다. C2 대비 C3의 SAM은3/3시드에서 악화했고 PSNR은2/3시드에서만 개선됐습니다. FR QNR은seed42·43에서 악화했지만 seed44는 +0.014235로 개선돼 시드별 방향도 달랐습니다. 원본 대비 C3의 QNR·Ds는3/3시드에서 악화했고 QNR 개선은60회 중3회뿐이었습니다. 정규화 이득과 W2 이득을 단순 합산할 수 없습니다.\n")
 add(f"요인 상호작용(C3−C2)−(C1−C0)의 평균은 PSNR {s['factorial_interaction']['RR']['PSNR']:+.6f}dB·QNR {s['factorial_interaction']['FR']['QNR']:+.6f}입니다. 이는 이 조건에서 두 변경의 결합이 단독 이득의 합보다 불리했다는 관측이며 FR 악화의 물리적 원인까지 입증하지는 않습니다. 원본 대비 개선을 논문 수치 대비 개선으로 해석하지 않습니다. 29·33·40번과 학습량·정밀도·샘플 순서가 완전히 같지 않으므로 기존 숫자를 이 실험의 대조군으로 대신 쓰지 않습니다.\n")
 add('## 5. 학습·평가 그래프\n')
 for name,label in [('train_mse','학습 MSE'),('val_mse','Validation MSE'),('val_psnr','Validation PSNR'),('val_sam','Validation SAM'),('comparison','전체 평가 비교'),('scene_deltas','장면별 차이')]:add(f'![{label}](results/figures/{name}.png)\n')
 add('## 6. 결과 이미지\n\n사전 고정한0기준 RR 장면1·8·11·13·19를 사용했습니다. LR MS nearest·PAN 입력·예측·GT 순서이며 RGB[2,1,0], MS 패널은 해당 장면 GT의1/99백분위 대비를 공유합니다. PAN은 별도 grayscale 대비입니다. 아래는 결합군 C3 seed42의 고정5장면이며 네 군·세 시드의 총60개 원본 패널도 모두 보존했습니다. test 결과로 장면·순서를 바꾸지 않았습니다. 지표에는 clipping·타일 분할·TTA를 사용하지 않았고 FR에는 GT를 만들지 않았습니다.\n')
 for index in cfg['scenes']:add(f'![C3 seed42 RR 장면{index}](results/C3_QB_s42/evaluation/RR_scene_{index:03}.png)\n')
 add('## 7. 재현 자료\n\n[현행 모델](model.py) · [실제 학습·평가 실행기](train.py) · [2슬롯 제어기](run.py) · [설정](config.json) · [검증·보고서 재생성](prepare.py) · [요약](results/summary.json) · [시드·장면 차이](results/paired_scene_deltas.json) · [보관 검증](results/verification.json) · [가중치](checkpoints) · [원본 source snapshots](provenance/source_snapshots) · [원본 실행 설정](provenance/runtime_config.json) · [초기 학습 설정](provenance/initial_runtime_config.json) · [데이터·캐시 해시](provenance/data_manifest.json) · [서버 export manifest](provenance/server_export_manifest.json) · [완료 기록](provenance/all_complete.json) · [최종 보존 manifest](manifest.json)\n')
 add(f"원본 initial·best·latest36개와 로그 수정 전 두 군의 best·latest4개, 총40개 체크포인트를 바이트 그대로 보존했습니다. 첫 에폭 initial 두 개는 최종 보관 initial과 바이트가 같아 중복 저장하지 않았습니다. 원본 export220개 파일의 해시를 확인했고40개 strict load·CPU FP32 합성 유한 추론,12개 latest의 optimizer step·내부 best 상태와 실제 best 일치를 검증했습니다. 전체150에폭의 샘플 순서와 validation 최소 선택,24개 전체20장 지표 평균과 평가 가중치 해시,60개 고정 이미지도 확인했습니다. 이는 원본 H5의 새로운 전체 재추론이나 장시간 CUDA 강제 종료 복구 시험과 다릅니다. 데이터·공유 정규화 캐시·대량 예측 배열은 Git에 추가하지 않았습니다.\n")
 add('```bash\npython SSA-MRN/experiments/performance/43_normalized_parallel_capacity/prepare.py\npython SSA-MRN/common/verify.py --research-track performance --experiment 43\n```\n')
 add('실제 서버 실행 폴더와 소스 경로는 provenance 원본 설정에 보존했습니다. 새 학습은 데이터 경로·runtime을 새 실행 위치로 명시적으로 구성하고 기존 완료 runtime을 덮어쓰지 않습니다. 자동 평가에는 validation best만 사용했으며 latest나 중간 에폭을 test로 재선택하지 않았습니다.\n')
 add('## 8. 개선 여부와 다음 단계\n\n**부분 개선·결합 후보 채택 보류**입니다. 정규화 단독의 평균 이득과 결합군의 RR·FR 상충을 분리합니다. 다음에는 결합군의 공간 왜곡이 커진 장면과 추가 가지의 기여를 분석한 뒤 새 가설을 정합니다. 현재 결과만으로 통계 증강·고주파 제한·게이트 추가 학습을 자동 실행하지 않습니다. 추가 센서·독립 holdout 및 MATLAB 수치 대응은 미실시이며 기존 원본 기준을 유지합니다.\n')
 (HERE/'README.md').write_text('\n'.join(lines),encoding='utf8')

def register(summary):
 name=HERE.name;delta=summary['contrasts']['C3_vs_C0'];text=f"3seed C3: PSNR {delta['RR']['PSNR']['mean_delta']:+.4f}dB·잠정 QNR {delta['FR']['QNR']['mean_delta']:+.5f}; C1 평균 개선·seed43 FR 악화, 결합 FR 완화 실패"
 entry=dict(report=f'experiments/performance/{name}/README.md',research_track='performance',experiment=43,display_name='43. MS 기준 가역 정규화와 저해상도 병렬 특징 용량 확장',experimenter='이병현',status='평가 완료',improvement='부분 개선',sensor='QB',trained_seeds=list(SEEDS),evaluated_seeds=list(SEEDS),test_scored=True,checkpoints=f'experiments/performance/{name}/checkpoints',results=f'experiments/performance/{name}/results',manifest=f'experiments/performance/{name}/manifest.json',config=f'experiments/performance/{name}/config.json',entrypoint=f'experiments/performance/{name}/run.py',verification=f'experiments/performance/{name}/prepare.py',limits='QNR 등 MATLAB 수치 대응 미검증; 기존 QB 탐색 benchmark; AMP skip으로 실제 업데이트 수 차이; 원본 기준 유지')
 entry['limits']='QNR 등 MATLAB 수치 대응 미검증; 기존 QB 탐색 benchmark; AMP skip으로 실제 업데이트 수 차이; 원본 기준 유지'
 for filename in ['experiments/catalog.json','research/experiments.json']:
  p=ROOT/'SSA-MRN'/filename;x=read(p);x['performance/'+name]=entry;write(p,x)
 for filename,prefix in [('README.md','SSA-MRN/experiments/performance/'),('SSA-MRN/README.md','experiments/performance/'),('SSA-MRN/research/performance/README.md','../../experiments/performance/')]:
  p=ROOT/filename;lines=p.read_text(encoding='utf8').splitlines();row=f'| [43. MS 기준 가역 정규화와 저해상도 병렬 특징 용량 확장]({prefix}{name}/README.md)<br>[이병현] | 정규화와 W2 가지를 네 군·세 시드로 비교했습니다. | {text} | 평가&nbsp;완료<br>3seed | 부분&nbsp;개선 |'
  matches=[i for i,x in enumerate(lines) if x.startswith('| [43.')]
  if matches:lines[matches[0]]=row
  else:
   idx=max(i for i,x in enumerate(lines) if x.startswith('| [42.'));lines.insert(idx+1,row)
  p.write_text('\n'.join(lines)+'\n',encoding='utf8')
def manifest():
 files={str(p.relative_to(HERE)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
 write(HERE/'manifest.json',dict(research_track='performance',experiment=43,experimenter='이병현',status='평가 완료',files=files));return len(files)
def main():
 p=argparse.ArgumentParser();p.add_argument('--skip-checkpoints',action='store_true',help='reuse prior checkpoint verification; does not claim a new strict load');p.add_argument('--register',action='store_true');a=p.parse_args()
 preservation,cfg=immutable_check();runs,s=collect(cfg)
 weights=read(HERE/'results/verification.json')['checkpoint_checks'] if a.skip_checkpoints else check_weights(runs)
 verification=dict(preservation=preservation,checkpoint_checks=weights,history_and_selection_checks=True,identical_epoch_orders=450,full_evaluations=24,scene_evaluations=480,fixed_images=60,precision='CPU FP32 verification; original training AMP FP16 and eval FP32',limit='hash/strict load/finite synthetic forward; no new full H5 evaluation')
 write(HERE/'results/verification.json',verification);figures(runs,s);report(runs,s,verification)
 if a.register:register(s)
 print('PASS',manifest(),'files;',weights['count'],'strict-load checkpoint records; C1 all-major-metrics all-seeds=',s['C1_all_major_metrics_all_seeds_improved'])
if __name__=='__main__':main()
