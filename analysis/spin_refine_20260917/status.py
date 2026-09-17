"""Read-only compact progress of the fixed refinement experiment."""
from pathlib import Path
import json,time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/spin-refine-20260917-v1'
rows=[]
for dataset in ('geant','abilene'):
 for variant in ('one_step','two_step'):
  for seed in (41001,41002):
   job=OUT/dataset/variant/f'seed{seed}'
   if not job.exists():continue
   row=dict(dataset=dataset,variant=variant,seed=seed)
   for name in ('status.json','smoke/status.json'):
    path=job/name
    if path.exists():
     data=json.loads(path.read_text());row['status' if name=='status.json' else 'smoke_status']={k:data.get(k) for k in ('state','epoch','epochs_completed','epochs_committed','best_epoch','best_score','error') if k in data}
   path=job/'smoke/SMOKE.json'
   if path.exists():
    data=json.loads(path.read_text());row['smoke']={k:data.get(k) for k in ('passed','exact_checkpoint_continuation','rough_epoch_seconds_from_smoke','parameter_count','peak_training_allocated_bytes')}
   path=job/'history.json'
   if path.exists():
    hist=json.loads(path.read_text());row['committed_epochs']=len(hist)
    if hist:
     row['latest']={k:hist[-1].get(k) for k in ('epoch','selection_score','best_score','best_epoch','training_seconds','evaluation_seconds')}
     tail=hist[-10:];row['mean_recent_epoch_seconds']=sum(x['training_seconds']+x['evaluation_seconds'] for x in tail)/len(tail)
   row['complete']=(job/'result.json').exists();rows.append(row)
failed=[]
for p in (OUT/'logs').glob('*.exit.json'):
 d=json.loads(p.read_text())
 if d['returncode']:
  failed.append(dict(record=str(p),returncode=d['returncode'],tail=Path(d['log']).read_text()[-5000:]))
print(json.dumps(dict(time_unix=time.time(),rows=rows,failed=failed),indent=2))
