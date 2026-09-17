"""Read-only compact progress; no scoring, models, or CUDA imports."""
from pathlib import Path
import json
import statistics
import time

root = Path(__file__).resolve().parents[2]
output = root / 'outputs/spin-path-controls-20260917-v1'
auth = json.loads((Path(__file__).with_name('AUTHORIZATION.json')).read_text())
report = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'allocation_remaining_minutes': round((auth['deadline_unix'] - time.time()) / 60, 1),
          'jobs': [], 'failures': []}
for path in sorted(output.glob('*/*/seed*/status.json')):
    status = json.loads(path.read_text())
    history_path = path.with_name('history.json')
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    recent = history[-5:]
    seconds = statistics.mean(row['training_seconds'] + row['evaluation_seconds']
                              for row in recent) if recent else None
    report['jobs'].append({'dataset': status['dataset'], 'variant': status['variant'],
        'seed': status['seed'], 'state': status['state'], 'saved': len(history),
        'latest_nmae': history[-1]['selection_score'] if history else None,
        'best_nmae': min(row['selection_score'] for row in history) if history else None,
        'recent_epoch_seconds': round(seconds, 2) if seconds else None,
        'estimated_remaining_minutes': round((160 - len(history)) * seconds / 60, 1) if seconds else None,
        'status_age_seconds': round(time.time() - path.stat().st_mtime, 1)})
for path in sorted(output.glob('logs/*.exit.json')):
    record = json.loads(path.read_text())
    if record['returncode']:
        report['failures'].append({**record, 'tail': Path(record['log']).read_text()[-2000:]})
print(json.dumps(report, separators=(',', ':')))
