"""Inventory tracked evidence without loading models or starting training."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path

def inventory(root):
    entries = subprocess.check_output(['git','ls-tree','-r','HEAD','--','SSA-MRN'],cwd=root,text=True).splitlines()
    tracked = {}
    for line in entries:
        header,path=line.split('\t',1);mode,kind,blob=header.split()
        if kind=='blob':tracked[path]=blob
    experiments={}
    for path,blob in tracked.items():
        match=re.match(r'(SSA-MRN/(?:research|experiments)/(?:performance|ssai|variable_scale)/\d+_[^/]+)/',path)
        if not match:continue
        base=match[1];row=experiments.setdefault(base,dict(path=base,checkpoints=[],evaluation_records=[],warnings=[]))
        if path.endswith('.pt'):row['checkpoints'].append(dict(path=path,git_blob=blob))
        if path.endswith('.json') and '/results/' in path and (root/path).exists():
            try: obj=json.loads((root/path).read_text(encoding='utf-8'))
            except (ValueError,UnicodeError):continue
            hits=[]
            def walk(x,keys=()):
                if isinstance(x,dict):
                    for k,v in x.items():
                        if k in ('PSNR','SAM','ERGAS','SCC','Q2n','QNR','D_lambda','D_s') and isinstance(v,(int,float)):
                            hits.append('.'.join((*keys,k)))
                        if isinstance(v,(dict,list)):walk(v,(*keys,k))
                elif isinstance(x,list):
                    for i,v in enumerate(x):
                        if isinstance(v,(dict,list)):walk(v,(*keys,str(i)))
            walk(obj)
            if hits:row['evaluation_records'].append(dict(path=path,git_blob=blob,metric_fields=hits))
        if path.endswith('README.md') and (root/path).exists():
            for n,line in enumerate((root/path).read_text(encoding='utf-8').splitlines(),1):
                if re.search(r'MATLAB|parity',line,re.I) and re.search(r'미검증|unverified|검증하지|검증되지',line,re.I):
                    row['warnings'].append(dict(path=path,line=n,text=line))
    for row in experiments.values():
        weights=row['checkpoints'];best=[x for x in weights if 'best' in Path(x['path']).name.lower()]
        covered={str(Path(x['path']).parent) for x in best}
        fallback=[x for x in weights if str(Path(x['path']).parent) not in covered and Path(x['path']).name=='latest.pt']
        row['candidate_checkpoints']=best+fallback
        row['status']='inventory_only_not_yet_reassessed'
    return dict(schema=1,scope='Tracked experiment evidence; candidates require original-contract reconciliation before execution',experiments=list(experiments.values()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=inventory(a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(experiments=len(result['experiments']),checkpoint_candidates=sum(len(x['candidate_checkpoints']) for x in result['experiments']),evaluation_records=sum(len(x['evaluation_records']) for x in result['experiments']))))
if __name__=='__main__':main()
