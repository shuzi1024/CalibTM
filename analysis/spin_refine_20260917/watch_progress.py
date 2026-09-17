"""Read-only live progress; exits after all four fixed trajectories finish."""
from pathlib import Path
import json,time,sys
ROOT=Path(__file__).resolve().parents[2]
dataset=sys.argv[1] if len(sys.argv)>1 else 'geant'
auth=json.loads((ROOT/'analysis/spin_refine_20260917/AUTHORIZATION.json').read_text())
while time.time()<auth['deadline_unix']:
 rows=[]
 for variant in ('one_step','two_step'):
  for seed in (41001,41002):
   job=ROOT/f'outputs/spin-refine-20260917-v1/{dataset}/{variant}/seed{seed}'
   row=dict(variant=variant,seed=seed,complete=(job/'result.json').exists())
   hp=job/'history.json'
   hist=json.loads(hp.read_text()) if hp.exists() else []
   row['epochs']=len(hist)
   if hist:
    row['best']=round(hist[-1]['best_score'],7)
    tail=hist[-10:]
    row['epoch_seconds']=round(sum(x['training_seconds']+x['evaluation_seconds'] for x in tail)/len(tail),2)
   sp=job/'status.json'
   if sp.exists():
    status=json.loads(sp.read_text());row['state']=status['state']
    if status['state'] in ('failed','stopped'):row['error']=status.get('error',status.get('reason'))
   rows.append(row)
 print(json.dumps(dict(time_unix=time.time(),dataset=dataset,rows=rows)),flush=True)
 if all(r['complete'] for r in rows):break
 if any(r.get('state') in ('failed','stopped') for r in rows):sys.exit(2)
 time.sleep(50)
