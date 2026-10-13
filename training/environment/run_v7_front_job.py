"""Run the front-face fine-tuning job, then export its test visualizations."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/train/MuJoCo_models/mujoco_v7_front_finetune'
STATUS = EXPERIMENT / 'job_status.json'


def status(state, **details):
    payload = dict(state=state, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), **details)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATUS)


def main():
    os.chdir(ROOT)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Host CUDA is unavailable; training has not started.')
        status('training', gpu=torch.cuda.get_device_name(0))
        subprocess.run([sys.executable, '-u', 'main.py', '--config',
                        'cfgs/MuJoCo_models/train.yaml', '--exp_name',
                        'mujoco_v7_front_finetune'], check=True)
        status('testing')
        subprocess.run([sys.executable, '-u', 'tools/visualize_mujoco_test.py',
                        '--experiment', str(EXPERIMENT)], check=True)
        output = max(EXPERIMENT.glob('test_vis_*'), key=lambda p: p.stat().st_mtime)
        with (output / 'metrics.csv').open() as stream:
            metrics = list(csv.DictReader(stream))
        samples = json.loads((ROOT / 'data/MuJoCoObject/dataset_v7_front/test.json').read_text())
        front_ids = {s['id'] for s in samples
                     if abs((s['pose']['azimuth_deg']+90) % 180-90) <= 24
                     and s['pose']['elevation_deg'] <= 25}
        front = [m for m in metrics if m['id'] in front_ids]
        keys = ['chamfer_l1_mm', 'gt_to_prediction_mean_mm', 'fscore_2mm']
        front_report = dict(samples=len(front), azimuth_tolerance_deg=24, max_elevation_deg=25,
                            means={k: sum(float(m[k]) for m in front)/len(front) for k in keys})
        (output / 'front_face_summary.json').write_text(json.dumps(front_report, indent=2))
        status('completed', output=str(output), checkpoint=str(EXPERIMENT / 'ckpt-best.pth'),
               front_face=front_report)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
