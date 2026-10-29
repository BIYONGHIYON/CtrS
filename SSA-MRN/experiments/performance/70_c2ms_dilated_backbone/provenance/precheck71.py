import pathlib,json,os,subprocess,sys
base=pathlib.Path('C:/CtrS-p70-71');p=base/'SSA-MRN/experiments/performance/71_coarse_pair_pan_blur'
cfg=json.loads((p/'config.json').read_text());dest=base/'precheck71';dest.mkdir(exist_ok=True);cfg['runtime']=str(dest)
cache=dest/'cache/WV3';cache.mkdir(parents=True,exist_ok=True)
for f in (base/'run70/cache/WV3').glob('*.npy'):
 d=cache/f.name
 if not d.exists():os.link(f,d)
cp=dest/'config.json';cp.write_text(json.dumps(cfg))
for mode in ('preflight','bench'):
 arms=cfg['arms'] if mode=='bench' else [None]
 for arm in arms:
  args=[sys.executable,'-u',str(p/'train.py'),'--config',str(cp),'--mode',mode]
  if arm:args+=['--sensor','WV3','--arm',arm,'--layout','channels_last','--tag','n1']
  subprocess.run(args,cwd=p,check=True)
print('71 FULL GPU PREFLIGHT AND BENCHMARK PASS',flush=True)
