"""Run the six authorized seed42 jobs; fails closed on any stage error."""
import os,json,sys,subprocess,shutil,hashlib
from pathlib import Path
BASE=Path('C:/CtrS-p70-71');PERF=BASE/'SSA-MRN/experiments/performance'
P70=PERF/'70_c2ms_dilated_backbone';P71=PERF/'71_coarse_pair_pan_blur'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONIOENCODING']='utf-8'
def command(folder,*args):
    subprocess.run([sys.executable,'-u',str(folder/'run.py'),*args],cwd=folder,check=True)
def main():
    for folder,num in ((P70,70),(P71,71)):
        runtime=BASE/f'run{num}'
        if not (runtime/'runtime_config.json').exists():command(folder,'--mode','prepare')
        if not (runtime/'preflight/optimization.json').exists():command(folder,'--mode','optimize')
        command(folder,'--mode','run','--resume')
        if num==70:
            for sensor in ('QB','WV3'):
                source=runtime/f'runs/{sensor}_C2MS_R_s42/checkpoints/best.pt'
                dest=P70/f'checkpoints/C2MS_R_{sensor}_s42/best.pt';dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists() and dest.read_bytes()!=source.read_bytes():raise RuntimeError('refusing to replace exported control')
                shutil.copy2(source,dest)
                assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(dest.read_bytes()).digest()
    (BASE/'suite_complete.json').write_text(json.dumps({'experiments':[70,71],'seeds':[42],'runs':6}))
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        (BASE/'suite_failure.txt').write_text(traceback.format_exc());raise
