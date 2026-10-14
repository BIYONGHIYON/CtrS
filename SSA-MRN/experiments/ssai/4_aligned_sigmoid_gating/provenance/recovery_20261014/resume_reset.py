from pathlib import Path
import json,sys,time
root=Path('C:/CtrS-ssai4/SSA-MRN');sys.path.insert(0,str(root))
from common.ssai_suite import write
run=Path('C:/CtrS-ssai4/runtime/qb_s42_100');backup=run/'recovery_20261014_1533'
record={'reason':'GPU calls and nvidia-smi became unresponsive around 12:44 KST; manual reboot confirmed; root cause unconfirmed','resume_epochs':{'A0':73,'A1':74},'source_plan_unchanged':True,'batch_precision_unchanged':True,'optimizer_rng_restored_by_original_runner':True,'checked_timestamp':time.time()}
# Saved metadata is already archived; only clear stale process identities after a confirmed reboot and no live suite process.
c=json.loads((run/'controller.json').read_text());c['pid']=0;c['recovery_pending']=True;write(run/'controller.json',c)
for arm in ('A0','A1'):
 s=json.loads((run/arm/'status.json').read_text());s['pid']=0;s['state']='resume_pending';s['updated']=time.time();write(run/arm/'status.json',s)
write(backup/'resume.json',record);print(json.dumps(record))
