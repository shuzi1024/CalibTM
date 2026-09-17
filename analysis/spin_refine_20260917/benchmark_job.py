"""Launch one deadline-bounded inference benchmark on an idle authorized GPU."""
from pathlib import Path
import json,os,socket,subprocess,time,sys
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'analysis/spin_refine_20260917'
dataset=sys.argv[1]
assert dataset in ('geant','abilene')
auth=json.loads((HERE/'AUTHORIZATION.json').read_text())
assert auth['state']=='authorized' and auth['started_unix']<=time.time()<auth['deadline_unix']-120
assert socket.gethostname() in auth['hosts']
for variant in ('one_step','two_step'):
 for seed in (41001,41002):
  result=json.loads((ROOT/f'outputs/spin-refine-20260917-v1/{dataset}/{variant}/seed{seed}/result.json').read_text())
  assert result['state']=='complete' and result['epochs_completed']==160
busy=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not busy,'Selected GPU is busy'
output=HERE/f'INFERENCE_BENCHMARK_{dataset.upper()}.json'
log=HERE/f'INFERENCE_BENCHMARK_{dataset.upper()}.log'
assert not output.exists() and not log.exists(),'Benchmark already exists; preserve it'
command=['timeout','--signal=TERM','--kill-after=10s',str(int(auth['deadline_unix']-time.time()-10))+'s',str(ROOT/'.venv/bin/python'),'-u',str(HERE/'benchmark_inference.py'),'--repo-root',str(ROOT),'--run-root','outputs/spin-refine-20260917-v1','--dataset',dataset,'--deadline-unix',str(auth['deadline_unix']),'--output',str(output)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with log.open('xb') as stream:
 proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(hostname=socket.gethostname(),gpu=0,timeout_pid=proc.pid,started_unix=time.time(),deadline_unix=auth['deadline_unix'],command=command,output=str(output),log=str(log))
(HERE/f'INFERENCE_BENCHMARK_{dataset.upper()}_LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
