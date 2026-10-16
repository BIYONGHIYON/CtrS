"""Verify archived original bytes and validation history; no training/test scoring."""
import argparse,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ARMS=('A0','A1','A2','A3')
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def write(p,x):(ROOT/p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8', newline="\n")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 mapping=read('provenance/retained_files.json')
 for old,meta in mapping.items():
  p=ROOT/meta['path']
  if not p.is_file() or p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']:raise ValueError('Original bytes changed: '+old)
 histories={a:read(f'results/{a}/history.json') for a in ARMS};best={};init=[];orders=[]
 decision=read('results/validation_decision.json')
 for a,h in histories.items():
  if [r['epoch'] for r in h]!=list(range(101)):raise ValueError('Incomplete history: '+a)
  if h[-1]['updates']!=26800 or h[-1]['skipped']!=0 or h[-1]['attempted']!=26800:raise ValueError('Update mismatch: '+a)
  selected=min(h,key=lambda r:r['val_mse']);best[a]={k:selected[k] for k in ('epoch','val_mse','val_psnr','val_sam')}
  if selected['epoch']!=decision['best'][a]['epoch'] or selected['val_mse']!=decision['best'][a]['val_mse']:raise ValueError('Best selection mismatch')
  ck=ROOT/'checkpoints'/f'{a}_QB_s42'
  for name in ('initial','best','latest'):
   if not (ck/(name+'.pt')).is_file():raise ValueError('Missing checkpoint')
  rec=read(f'results/{a}/training_complete.json')
  if sha(ck/'best.pt')!=decision['best'][a]['best_sha256'] or sha(ck/'latest.pt')!=rec['latest_sha256'] or sha(ck/'best.pt')!=rec['best_sha256']:raise ValueError('Checkpoint record mismatch')
  init.append(read(f'results/{a}/initialization.json')['state_sha256']);orders.append([r['order_sha256'] for r in h[1:]])
  pf=read(f'results/{a}/preflight.json')
  if not all(pf[k] for k in ('forward','backward','gpu_cache')) or not pf['cuda_resume_equivalence']['passed']:raise ValueError('GPU preflight record failed')
 if len(set(init))!=1 or any(o!=orders[0] for o in orders):raise ValueError('Common initialization/order mismatch')
 repeat={a:best[a]['val_mse']<=best['A3']['val_mse']*1.005 for a in ('A1','A2')}
 if repeat!=decision['simplified_arm_pass'] or any(repeat.values())!=decision['repeat_all_four_arms_seeds43_44']:raise ValueError('Repeat gate mismatch')
 summary={'best_validation':best,'relative_MSE_percent_vs_A0':{a:100*(r['val_mse']/best['A0']['val_mse']-1) for a,r in best.items()},'relative_MSE_percent_vs_A3':{a:100*(r['val_mse']/best['A3']['val_mse']-1) for a,r in best.items()},'repeat_seeds43_44':any(repeat.values()),'test_scored':False,'status':'학습 완료·평가 대기'}
 write('results/summary.json',summary)
 write('results/retention_verification.json',{'original_retained_files':len(mapping),'original_retained_bytes_match':True,'checkpoints':12,'each_arm_completed_epochs':100,'each_arm_updates':26800,'skipped_updates':0,'initial_state_hashes_equal':True,'actual_epoch_orders_equal':True,'validation_best_and_repeat_gate_recomputed':True,'source_gpu_preflight_records_pass':True,'archive_environment_checkpoint_strict_load':False,'archive_environment_original_H5_reinference':False,'test_scored':False})
 print(f'PASS: {len(mapping)} retained original hashes; 12 checkpoints; four 100-epoch histories; initialization/order/best/repeat gate')
 return summary

def report(s):
 best=s['best_validation'];rows=[]
 names={'A0':'원본','A1':'평균 제거','A2':'대비 조절','A3':'전체 정규화'}
 for a in ARMS:
  r=best[a];rows.append(f"| {a} {names[a]} | {r['epoch']} | {r['val_mse']:.12f} | {r['val_psnr']:.6f} | {r['val_sam']:.6f} | {s['relative_MSE_percent_vs_A0'][a]:+.3f}% | {s['relative_MSE_percent_vs_A3'][a]:+.3f}% |")
 text='''# MS 기준 평균 제거와 대비 정규화의 기여 분리

**연구 분야:** 기존 성능 개선

**실험자:** 김경찬

**변경 내용:** 원본 SSA-MRN 앞뒤의 샘플별 공유 정규화에서 평균 제거와 대비 조절을 분리해 네 군으로 비교했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|:---:|:---:|
| seed42 네 군의100epoch 학습을 완료했습니다. A3 validation MSE가 가장 낮았고 A1·A2는 반복 진행 기준을 넘었습니다. RR·FR은 미평가입니다. | 학습&nbsp;완료<br>평가&nbsp;대기 | 판단&nbsp;보류 |

[실험 목록](../../../research/performance/README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [재현 자료](#evidence)

## 1. 목적과 상태

33·38번의 공유 가역 정규화는 RR 이득이 반복됐으나 FR은 시드별로 엇갈렸습니다. 39번 국소 정규화도 전역 방식보다 평균 성능이 악화했습니다. 이번에는 전역 정규화의 평균 제거와 대비 조절을 분리해 어떤 요소가 validation 이득에 기여하는지 확인했습니다.

seed42 네 군 모두100epoch·26,800회 업데이트를 완료했습니다. 학습과 validation은 완료했으며 RR·FR test는 의도적으로 실행하지 않았습니다. 저장된 가중치를 이용한 전체 평가가 다음 단계입니다. 최종 성능 개선이나 새 기준 모델 채택을 확정하지 않습니다.

## 2. 변경한 구조와 비교 조건

LR MS 전체 C·H·W에서 샘플별 모집단 평균 μ와 표준편차 σ를 구하고 s=max(σ,0.01)로 설정합니다. GT와 다른 배치 샘플은 통계 계산에 포함하지 않습니다. PAN·제공 LMS·MS에 동일한 스칼라 변환을 적용합니다.

| 군 | 입력 변환 | 출력 역변환 |
|---|---|---|
| A0 원본 | x | 없음 |
| A1 평균 제거 | x−μ | y+μ |
| A2 대비 조절 | x/s | sy |
| A3 전체 정규화 | (x−μ)/s | sy+μ |

원본 초기 LRPAN·backbone BatchNorm·밴드별 SSAI·다중 해상도 복원을 유지했습니다. 추가 학습 파라미터는 없습니다. 네 군은 같은 초기 파라미터를 사용하지만 입력 변환 때문에 초기 출력은 다릅니다.

| 항목 | 실제 공통 조건 |
|---|---|
| 데이터 | QuickBird train17,139·validation1,905·RR20·FR20; test는 미사용 |
| 특징 차원·공간 배율 | K4·4배 |
| 학습 | fresh seed42·각100epoch·Adam·고정lr1e-4 |
| 배치 | effective64·micro32, 마지막 불완전 배치 포함 |
| 손실 | 역변환한 최종 출력의 원래 DN/2047 공간 MSE |
| 정밀도 | 학습FP16 AMP; 통계·변환·역변환·손실·validation FP32; TF32 off |
| 장치 | Tesla T4×2; 첫 묶음 GPU0 A0·GPU1 A3, 다음 묶음 GPU0 A1·GPU1 A2 |
| best | epoch0 포함 전체 validation MSE 최소, 정확한 동률이면 이른 epoch |
| 공정성 | 네 군 초기 상태와 실제100epoch 배치 순서 해시 일치; 업데이트 생략0회 |

원래 DN/2047 정규화와 제공 LMS를 유지하며 입력·예측을 clipping하지 않습니다. micro batch도 고정해 BatchNorm 조건을 맞췄습니다. 학습된 가중치나 checkpoint 내부 설정을 정리를 위해 변경하지 않았습니다.

<a id="results"></a>

## 3. 정량 결과

아래는 전체 validation 분할의 각 군 best입니다. MSE 변화율은 음수일수록 좋습니다. 이 값은 RR·FR test 결과가 아닙니다.

| 군 | best epoch | validation MSE ↓ | validation PSNR ↑ | validation SAM ↓ | MSE 변화 vs A0 | MSE 변화 vs A3 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

[전체 epoch 기록](results/) · [집계 수치](results/summary.json) · [학습 전 고정한 선택 규칙과 실제 결정](results/validation_decision.json)

RR PSNR·SAM·ERGAS·SCC·Q2n과 FR QNR·Dλ·Ds는 아직 계산하지 않았습니다. QNR·Q2n·SCC의 공식 수치 대응도 별도 검증이 필요합니다. FR에는 고해상도 GT가 없습니다.

## 4. 기준선 대비 차이와 해석

A1·A2는 원본 A0보다 validation MSE가 각각1.421%·0.847% 낮지만 A3보다1.220%·1.810% 높습니다. A3는 A0보다2.610% 낮습니다. 현재 조건에서는 두 연산을 함께 유지하는 구성이 validation에서 가장 유리했습니다.

사전 규칙은 A1 또는 A2의 best validation MSE가 A3의1.005배 이내이면 네 군을 seed43·44로 반복하는 것이었습니다. 둘 다 이 범위를 벗어나 반복 확대를 중단합니다. 이 결정에 test를 사용하지 않았습니다. A0·A1 best가 마지막epoch이므로 충분히 수렴한 뒤에도 같은 순위가 유지된다고 확정하지 않습니다. 단일 시드의 결과를 정규화 효과의 보편적 인과 증거로 해석하지 않습니다.

<a id="graphs"></a>

## 5. 학습·평가 그래프

![실제 네 군의 학습 MSE와 validation MSE·PSNR·SAM](results/curves.png)

실제100epoch 기록을 사용한 원본 곡선입니다. validation 곡선에는epoch0을 포함하며 최종 구간의 작은 차이는 원본 숫자와 함께 확인합니다. test 기준선 비교 그래프는 미평가 상태이므로 만들지 않았습니다.

<a id="images"></a>

## 6. 결과 이미지

예측 이미지는 아직 생성하지 않았습니다. 학습 전에 RR의0기준 전체 장면1·8·11·13·19를 이 순서로 고정했습니다. 평가 시 LR MS 최근접 확대·PAN 입력·예측·GT의4패널로 저장하며 RGB[2,1,0]과 동일 장면 GT의1/99백분위 대비를 MS 패널에 공유합니다. PAN은 별도 grayscale 대비를 사용합니다. FR에는 GT를 만들지 않습니다.

<a id="evidence"></a>

## 7. 재현 자료

[단일 셀 실행기](run.py) · [보관 검증·집계](prepare.py) · [보관 설정](config.json) · [학습 당시 설정](provenance/config.json) · [원본 실행 셀](provenance/run_cell.py) · [원본 소스](provenance/source/) · [가중치12개](checkpoints/) · [데이터 manifest](provenance/data_manifest.json) · [예시 선택](provenance/selection.json) · [원본 보존 대응](provenance/retained_files.json) · [보존 검증](results/retention_verification.json) · [manifest](manifest.json)

각 군의 실제 initial·best·latest를 원본 바이트 그대로 보존했습니다. latest에는 optimizer·AMP scaler·Python/NumPy/Torch CPU/CUDA RNG·배치 generator·현재 순서·다음 batch·누적 손실·history·best_model이 포함된다는 실행 기록을 보존했습니다. 원본 소스·설정·compact epoch 기록과 원본 학습 곡선을 함께 보관합니다.

보관 환경에서 원본 ZIP SHA256·CRC·내부63개 항목 해시를 확인했고 그중 상세 로그·상태12개를 제외한 원본51개를 이 폴더에 바이트 그대로 보존했습니다. 네 군의101개 epoch 기록·공통 초기 상태·실제 배치 순서·업데이트 수·validation best·반복 진행 규칙을 독립 재계산했습니다. Kaggle에서 네 군의 유한 forward/backward·GPU 상주·두 업데이트 AMP 재개 검사가 통과한 기록을 보존했습니다. 이는 장시간 전체 세션 강제 종료 복구를 검증한 결과와 다릅니다.

이 보관 환경에는 PyTorch·GPU·원본 H5가 없어 checkpoint strict load·원본 데이터 재추론·test 성능 재평가는 수행하지 않았습니다. 해시 검증을 추론 검증으로 표시하지 않습니다. 원본 runtime과 보관 폴더의 파일 배치는 다르므로 원본 Saved Output 또는 다운로드한 training ZIP을 전체 복원해 평가를 재개합니다. 안내는 [학습 당시 실행 문서](provenance/source/KAGGLE_GUIDE.md)를 따릅니다.

```bash
python SSA-MRN/experiments/performance/40_ms_center_scale_ablation/prepare.py
python SSA-MRN/common/verify.py --research-track performance --experiment 40
```

## 8. 개선 여부와 다음 단계

**학습 완료·평가 대기 / 판단 보류**입니다. 평균 제거·대비 조절을 분리한 두 후보는 사전 validation 기준을 넘었으므로 seed43·44 반복은 진행하지 않습니다. 저장된 네 군의 best로 RR·FR 각20장 전체 평가를 수행하고 고정5장면·전체 숫자·비교 그래프를 보존합니다. test를 본 뒤 best·학습량·정규화 설정을 바꾸지 않습니다.
'''
 (ROOT/'README.md').write_text(text,encoding='utf-8', newline="\n")

def register():
 repo=ROOT.parents[3];ssa=repo/'SSA-MRN';key='performance/'+ROOT.name
 item=dict(report='experiments/performance/'+ROOT.name+'/README.md',research_track='performance',experiment=40,display_name='40. MS 기준 평균 제거와 대비 정규화의 기여 분리',experimenter='김경찬',status='학습 완료·평가 대기',improvement='판단 보류',sensor='QB',trained_seeds=[42],evaluated_seeds=[],test_scored=False,checkpoints='experiments/performance/'+ROOT.name+'/checkpoints',results='experiments/performance/'+ROOT.name+'/results',manifest='experiments/performance/'+ROOT.name+'/manifest.json',config='experiments/performance/'+ROOT.name+'/config.json',entrypoint='experiments/performance/'+ROOT.name+'/run.py',verification='experiments/performance/'+ROOT.name+'/prepare.py',limits='seed42 validation만 확인. A1·A2의A3 대비MSE +1.220%·+1.810%로 사전 반복 기준 미달. RR·FR 미평가, 최종 개선 판단 보류.')
 for p in [ssa/'experiments/catalog.json',ssa/'research/experiments.json']:
  d=json.loads(p.read_text());d[key]=item;p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n', newline="\n")
 for p,link in [(repo/'README.md','SSA-MRN/'),(ssa/'README.md',''),(ssa/'research/performance/README.md','../../')]:
  s=p.read_text();row=f'| [40. MS 기준 평균 제거와 대비 정규화의 기여 분리]({link}experiments/performance/{ROOT.name}/README.md)<br>[김경찬] | 전역 공유 정규화의 평균 제거·대비 조절을 네 군으로 분리했습니다. | seed42 validation: A3 MSE 최저; A1·A2는A3 대비 +1.220%·+1.810%로 반복 중단. RR·FR 미평가 | 학습&nbsp;완료<br>평가&nbsp;대기 | 판단&nbsp;보류 |'
  lines=s.splitlines();found=[i for i,x in enumerate(lines) if x.startswith('| [40.')]
  if found:lines[found[0]]=row
  else:
   idx=max(i for i,x in enumerate(lines) if x.startswith('| [') and ('performance/' in x or p.name=='README.md' and p.parent.name=='performance'))
   lines.insert(idx+1,row)
  p.write_text('\n'.join(lines)+'\n', newline="\n")
 print('Five experiment indexes updated')

def manifest():
 files={}
 for p in sorted(ROOT.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='manifest.json':files[p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
 write('manifest.json',{'research_track':'performance','experiment':40,'experimenter':'김경찬','status':'학습 완료·평가 대기','test_scored':False,'files':files})
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');args=p.parse_args()
 s=verify();report(s)
 if args.register:register()
 manifest();print('Report, validation summary and manifest generated; RR/FR NOT evaluated.')
