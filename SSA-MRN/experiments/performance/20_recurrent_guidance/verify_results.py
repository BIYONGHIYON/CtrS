"""Verify recorded D0-D6 training and D0/D1/D3/D4/D5/D6 full test evaluation."""
import sys,json,math,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from common.verify import verify

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-torch',action='store_true');a=p.parse_args()
    folder=Path(__file__).resolve().parent;verify(folder);r=folder/'results';selected={}
    for v in ('D0','D1','D2','D3','D4','D5','D6'):
        history=read(r/v/'history.json');assert [x['epoch'] for x in history]==list(range(1,51))
        assert all(x['skipped_steps']==0 for x in history)
        selected[v]=min(history,key=lambda x:(x['val_mse'],x['val_sam'],x['epoch']))
    reports=[read(r/'evaluation/metrics.json'),read(r/'additional_evaluation/metrics.json')]
    seen=set()
    for report in reports:
        for v,item in report['runs'].items():
            assert v not in seen;seen.add(v)
            path=folder/'checkpoints'/v/'best.pt'
            assert item['best_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            assert item['best_epoch']==selected[v]['epoch']
            for protocol in ('RR','FR'):
                record=item['metrics'][protocol];assert [x['index'] for x in record['rows']]==list(range(20))
                for k,mean in record['mean'].items():
                    values=[x['metrics'][k] for x in record['rows']]
                    assert all(math.isfinite(x) for x in values)
                    assert math.isclose(sum(values)/20,mean,rel_tol=1e-12,abs_tol=1e-14)
    assert seen=={'D0','D1','D3','D4','D5','D6'}
    if a.check_torch:
        import torch
        from common.checkpoint import load_model,predict
        torch.set_num_threads(1)
        for path in (folder/'checkpoints').rglob('*.pt'):
            model,ck=load_model(20,path)
            with torch.inference_mode():y=predict(20,model,torch.ones(1,1,64,64),torch.ones(1,4,64,64),torch.ones(1,4,16,16))
            assert torch.isfinite(y).all()
    print('PASS: 350 training epochs; six candidates, 240 RR/FR rows; unchanged validation selection; D2 test unavailable')
if __name__=='__main__':main()
