"""Finish one authorized phase: wait, measure inference costs, build its report.

No new training is launched here. Existing frozen trajectories are read only.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'analysis/spin_refine_20260917'
OUT=ROOT/'outputs/spin-refine-20260917-v1'

def write(path, value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def work(dataset,auth):
    state_path=HERE/f'PHASE_{dataset.upper()}.json'
    start=time.time()
    def status(state,**extra):
        value=dict(dataset=dataset,state=state,started_unix=start,updated_unix=time.time(),deadline_unix=auth['deadline_unix'],**extra)
        write(state_path,value)
        print(json.dumps(value),flush=True)
    def check():
        if time.time()>=auth['deadline_unix']-90:
            raise TimeoutError('Authorization deadline reached before phase completion')
    with (HERE/f'finish-{dataset}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                check()
                done=0
                for variant in ('one_step','two_step'):
                    for seed in (41001,41002):
                        job=OUT/dataset/variant/f'seed{seed}'
                        sp=job/'status.json'
                        if sp.exists():
                            s=json.loads(sp.read_text())
                            if s['state'] in ('failed','stopped'):
                                raise RuntimeError(f'{variant}/{seed}: '+str(s))
                        rp=job/'result.json'
                        if rp.exists():
                            r=json.loads(rp.read_text())
                            assert r['state']=='complete' and r['epochs_completed']==160
                            done+=1
                if done==4:break
                status('waiting_for_training',complete=done,total=4)
                time.sleep(30)
            status('waiting_for_idle_benchmark_gpu',complete=4,total=4)
            while subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                check();time.sleep(5)
            output=HERE/f'INFERENCE_BENCHMARK_{dataset.upper()}.json'
            launch=HERE/f'INFERENCE_BENCHMARK_{dataset.upper()}_LAUNCH.json'
            if not output.exists() and not launch.exists():
                check()
                subprocess.run([sys.executable,str(HERE/'benchmark_job.py'),dataset],check=True,cwd=ROOT)
            status('measuring_inference_cost',complete=4,total=4)
            while True:
                check()
                if output.exists():
                    b=json.loads(output.read_text())
                    if b['state']=='complete':break
                    if b['state'] in ('failed','stopped'):
                        raise RuntimeError('Benchmark did not finish: '+str(b.get('error',b.get('reason'))))
                time.sleep(5)
            check()
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable,str(HERE/'report_refinement.py'),'--dataset',dataset,'--cost-reference',str(output),'--plot-curves'],check=True,cwd=ROOT,env=environment)
            summary=HERE/f'report_{dataset}/SUMMARY.json'
            record=json.loads(summary.read_text())
            assert record['state']=='complete'
            status('complete',complete=4,total=4,summary=str(summary),summary_sha256=sha(summary),component_screen_passed=record['component_screen_passed'],eligible_to_consider_further_validation=record['eligible_to_consider_further_validation'],automatic_training_expansion=False)
        except BaseException as error:
            status('failed',error=f'{type(error).__name__}: {error}')
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('geant','abilene'),required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    auth=json.loads((HERE/'AUTHORIZATION.json').read_text())
    assert auth['state']=='authorized' and auth['started_unix']<=time.time()<auth['deadline_unix']-180
    assert socket.gethostname()==auth['hosts'][0], 'Cost measurement must use the same host'
    if args.worker:return work(args.dataset,auth)
    log=HERE/f'finish-{args.dataset}.log'
    assert not log.exists(), 'Existing phase controller must be inspected before relaunch'
    command=['timeout','--signal=TERM','--kill-after=10s',str(int(auth['deadline_unix']-time.time()-10))+'s',sys.executable,'-u',str(Path(__file__).resolve()),'--dataset',args.dataset,'--worker']
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(dataset=args.dataset,hostname=socket.gethostname(),pid=proc.pid,started_unix=time.time(),deadline_unix=auth['deadline_unix'],command=command,log=str(log),files_sha256={name:sha(HERE/name) for name in ('finish_phase.py','benchmark_job.py','benchmark_inference.py','report_refinement.py')})
    write(HERE/f'FINISH_{args.dataset.upper()}_LAUNCH.json',record)
    print(json.dumps(record))

if __name__=='__main__':main()
