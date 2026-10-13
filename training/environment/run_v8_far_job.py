"""Run far-distance fine-tuning and summarize test errors by distance."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/train/MuJoCo_models/mujoco_v8_far_finetune'
STATUS = EXPERIMENT / 'job_status.json'


def status(state, **details):
    payload = dict(state=state, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), **details)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATUS)


def aggregate(rows):
    keys = ['chamfer_l1_mm', 'gt_to_prediction_mean_mm', 'fscore_2mm']
    return dict(samples=len(rows), means={k: sum(float(r[k]) for r in rows)/len(rows)
                                         for k in keys} if rows else {})


def main():
    os.chdir(ROOT)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Host CUDA is unavailable; training has not started.')
        status('training', gpu=torch.cuda.get_device_name(0))
        subprocess.run([sys.executable, '-u', 'main.py', '--config',
                        'cfgs/MuJoCo_models/train.yaml', '--exp_name',
                        'mujoco_v8_far_finetune'], check=True)
        status('testing')
        subprocess.run([sys.executable, '-u', 'tools/visualize_mujoco_test.py',
                        '--experiment', str(EXPERIMENT)], check=True)
        output = max(EXPERIMENT.glob('test_vis_*'), key=lambda p: p.stat().st_mtime)
        with (output / 'metrics.csv').open() as stream:
            metrics = list(csv.DictReader(stream))
        samples = json.loads((ROOT / 'data/MuJoCoObject/dataset_v8_far/test.json').read_text())
        poses = {s['id']: s['pose'] for s in samples}
        distances = sorted({p['distance_m'] for p in poses.values()})
        report = dict(
            near=aggregate([m for m in metrics if poses[m['id']]['distance_m'] <= 1.3]),
            far=aggregate([m for m in metrics if poses[m['id']]['distance_m'] >= 1.5]),
            by_distance_m={str(d): aggregate([m for m in metrics if poses[m['id']]['distance_m'] == d])
                           for d in distances},
            front_face=aggregate([m for m in metrics
                                  if abs((poses[m['id']]['azimuth_deg']+90) % 180-90) <= 24
                                  and poses[m['id']]['elevation_deg'] <= 25]))
        (output / 'distance_summary.json').write_text(json.dumps(report, indent=2))
        status('completed', output=str(output), checkpoint=str(EXPERIMENT / 'ckpt-best.pth'),
               near=report['near'], far=report['far'])
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
