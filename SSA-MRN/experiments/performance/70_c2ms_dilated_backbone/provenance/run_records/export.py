import pathlib,json,hashlib,tarfile
B=pathlib.Path('C:/CtrS-p70-71');out=B/'export70_71';out.mkdir(exist_ok=False)
assert (B/'suite_complete.json').exists()
records={}
for num,name in [(70,'70_c2ms_dilated_backbone'),(71,'71_coarse_pair_pan_blur')]:
 r=B/f'run{num}';cfg=json.loads((r/'runtime_config.json').read_text(encoding='utf-8'));dest=out/name;dest.mkdir()
 def save(src,rel):
  p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(src.read_bytes());records[str(p.relative_to(out))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 for sensor,arm,seed in cfg['queue']:
  run=r/'runs'/f'{sensor}_{arm}_s{seed}';done=json.loads((run/'training_complete.json').read_text(encoding='utf-8'));ev=json.loads((run/'evaluation_complete.json').read_text(encoding='utf-8'))
  assert done['epoch']==150 and done['fingerprint']==cfg['fingerprint']
  assert hashlib.sha256((run/'checkpoints/best.pt').read_bytes()).hexdigest()==done['best_sha256']
  for f in (run/'checkpoints').glob('*.pt'):save(f,pathlib.Path('checkpoints')/run.name/f.name)
  for f in run.rglob('*'):
   if f.is_file() and f.suffix in ('.json','.png') and 'checkpoints' not in f.parts:save(f,pathlib.Path('results')/run.name/f.relative_to(run))
 for f in r.glob('*.json'):save(f,pathlib.Path('provenance/run_records')/f.name)
 for f in (r/'preflight').glob('*.json'):save(f,pathlib.Path('provenance/run_records/preflight')/f.name)
for f in ('suite_complete.json','recovery_launcher.json'):
 records[f]={'note':(B/f).read_text(encoding='utf-8-sig')}
(out/'export_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
with tarfile.open(B/'export70_71.tgz','w:gz') as t:t.add(out,arcname='export70_71')
print('EXPORT PASS',len(records),(B/'export70_71.tgz').stat().st_size,flush=True)
