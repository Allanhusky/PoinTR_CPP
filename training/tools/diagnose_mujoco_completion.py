"""Offline audit of paired data and saved PoinTr folding patches (no GPU needed)."""
import argparse
import json
from pathlib import Path
import time
import sys

import numpy as np
from scipy.spatial import cKDTree
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit_dataset(root):
    groups, identities, counts, errors, unique = [], set(), {}, [], []
    for split in ['train', 'val', 'test']:
        rows = json.loads((root / f'{split}.json').read_text())
        counts[split] = len(rows)
        groups.append({r['group'] for r in rows})
        for row in rows:
            if row['id'] in identities:
                raise ValueError('Sample ID occurs in multiple splits')
            identities.add(row['id'])
            n, meta = row['normalization'], row['source_metadata']
            transform = np.asarray(meta['T_world_camera_optical'])
            for key, raw_name in [('partial', 'partial_object.npy'), ('complete', 'complete_object.npy')]:
                xyz = np.load(root / row[key], allow_pickle=False)
                if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
                    raise ValueError(f'Invalid {key}: {row["id"]}')
                restored = xyz*n['radius_m'] + np.asarray(n['centroid'])
                if n['frame'] == 'camera':
                    restored = restored @ transform[:3, :3].T + transform[:3, 3] - meta['object_center_world_m']
                elif n['frame'] != 'object':
                    raise ValueError(f'Unknown coordinate frame: {n["frame"]}')
                raw = np.load(Path(row['source']) / raw_name, allow_pickle=False)
                errors.append(float(cKDTree(raw).query(restored[::64])[0].max()))
                if key == 'partial':
                    unique.append((len(np.unique(xyz, axis=0)), len(xyz)))
    if any(not groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError('Azimuth sectors overlap between splits')
    if max(errors) > 1e-6:
        raise ValueError('Input/truth inverse transforms do not match their source points')
    return dict(counts=counts, frame=n['frame'], max_inverse_error_m=max(errors),
                minimum_unique_input_points=min(value for value, count in unique),
                samples_with_repeated_input=sum(value < count for value, count in unique),
                split_sector_overlap=False)


def audit_experiment(experiment):
    config = yaml.safe_load((experiment / 'config.yaml').read_text())
    folders = [p for p in experiment.glob('test_vis_*') if (p / 'summary.json').is_file()]
    if not folders:
        raise ValueError(f'No saved test output in {experiment}')
    results = max(folders, key=lambda p: p.stat().st_mtime)
    summary = json.loads((results / 'summary.json').read_text())
    model = config['model']
    queries = model['num_query']
    side = int(np.sqrt(model['num_pred']//queries)+.5)
    generated_count = queries*side**2
    ratios, per_sample = [], []
    for filename in sorted(results.glob('*/prediction_normalized.npy')):
        prediction = np.load(filename)
        metadata = json.loads((filename.parent / 'metadata.json').read_text())
        row = metadata['sample']
        dataset = Path(summary['dataset'])
        partial = np.load(dataset / row['partial'])
        if prediction.shape != (generated_count+len(partial), 3) or not np.isfinite(prediction).all():
            raise ValueError(f'Invalid saved prediction: {filename}')
        # PoinTr concatenates generated patches first, then the untouched input.
        if not np.allclose(prediction[generated_count:], partial, atol=1e-6):
            raise ValueError(f'Unexpected output ordering: {filename}')
        patches = prediction[:generated_count].reshape(queries, side**2, 3)
        centered = patches-patches.mean(axis=1, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        ratio = singular[:, 1]/np.maximum(singular[:, 0], 1e-10)
        ratios.extend(ratio.tolist())
        per_sample.append(dict(id=row['id'], line_like_patch_fraction=float((ratio < .1).mean()),
                               chamfer_l1_mm=metadata['metrics']['chamfer_l1_mm']))
    ratios = np.asarray(ratios)
    dataset_report = audit_dataset(Path(summary['dataset']))
    steps = dataset_report['counts']['train']//config['total_bs']
    report = dict(experiment=str(experiment), saved_test=str(results), checkpoint=summary['checkpoint'],
        dataset=dataset_report, batch_size=config['total_bs'], learning_rate=config['optimizer']['kwargs']['lr'],
        epochs=config['max_epoch']+1, batches_per_epoch=steps,
        estimated_optimizer_updates=steps*(config['max_epoch']+1)//config['step_per_update'],
        discarded_samples_per_epoch=dataset_report['counts']['train'] % config['total_bs'],
        queries=queries, folding_grid=[side, side], generated_points=generated_count,
        patches_analyzed=len(ratios), median_patch_s2_over_s1=float(np.median(ratios)),
        line_like_patch_fraction=float((ratios < .1).mean()),
        line_like_definition='second/first PCA singular value < 0.1; geometric heuristic, not a universal quality metric',
        chamfer_l1_mm=summary['means']['chamfer_l1_mm'], per_sample=per_sample)
    return report


def plot_patches(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = sorted(report['per_sample'], key=lambda r: r['chamfer_l1_mm'])
    chosen = rows[len(rows)//2]
    folder = Path(report['saved_test']) / chosen['id']
    meta = json.loads((folder / 'metadata.json').read_text())
    n = meta['sample']['normalization']
    prediction = np.load(folder / 'prediction_normalized.npy')
    clouds = [np.load(folder / f'{name}_m.npy')*1000 for name in ['input', 'prediction', 'ground_truth']]
    patches = (prediction[:report['generated_points']]*n['radius_m']+np.asarray(n['centroid']))*1000
    patches = patches.reshape(report['queries'], -1, 3)
    fig = plt.figure(figsize=(16, 4.5), layout='constrained')
    joined = np.concatenate(clouds)
    center = (joined.max(0)+joined.min(0))/2
    half = np.ptp(joined, axis=0).max()*.55
    for col, (title, cloud) in enumerate([('Input', clouds[0]), ('Generated only: colors = query patches', patches.reshape(-1, 3)), ('Ground truth', clouds[2])]):
        ax = fig.add_subplot(1, 4, col+1, projection='3d')
        color = np.repeat(np.arange(report['queries']), patches.shape[1]) % 20 if col == 1 else '#2980b9'
        ax.scatter(*cloud.T, s=1, c=color, cmap='tab20' if col == 1 else None, alpha=.75)
        ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
               zlim=(center[2]-half, center[2]+half), xlabel='X mm', ylabel='Y mm', zlabel='Z mm', title=title)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=40)
    centered = patches-patches.mean(1, keepdims=True)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    ratios = singular[:, 1]/np.maximum(singular[:, 0], 1e-10)
    index = int(np.argmin(ratios))
    xy = centered[index] @ axes[index, :2].T
    side = report['folding_grid'][0]
    xy = xy.reshape(side, side, 2)
    ax = fig.add_subplot(1, 4, 4)
    for row in xy:
        ax.plot(*row.T, color='#2980b9', linewidth=.6)
    for row in xy.transpose(1, 0, 2):
        ax.plot(*row.T, color='#e67e22', linewidth=.6)
    ax.scatter(*xy.reshape(-1, 2).T, s=8, color='black')
    ax.set_aspect('equal')
    ax.set(xlabel='Patch principal axis 1 (mm)', ylabel='Axis 2 (mm)',
           title=f'Most elongated patch: {side}x{side}\nS2/S1={ratios[index]:.3f}')
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def gpu_audit(report):
    """Forward only; seed substitution lives in memory and never edits weights."""
    import torch
    from tools import builder
    from utils.config import cfg_from_yaml_file
    from extensions.chamfer_dist import ChamferDistanceL1
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA unavailable')
    torch.set_num_threads(2)
    config = cfg_from_yaml_file(str(Path(report['experiment'])/'config.yaml'))
    model = builder.model_builder(config.model).cuda().eval()
    builder.load_model(model, report['checkpoint'])
    saved_seed = model.foldingnet.folding_seed.clone()
    dataset = Path(config.dataset.test._base_.DATA_PATH)
    criterion = ChamferDistanceL1()
    result = {}
    try:
        with torch.inference_mode():
            for variant, split in [('regular', 'train'), ('regular', 'val'), ('regular', 'test'), ('random_seed', 'test')]:
                if variant == 'random_seed':
                    random = np.random.default_rng(42).uniform(-1, 1, saved_seed.shape).astype('float32')
                    model.foldingnet.folding_seed = torch.from_numpy(random).cuda()
                else:
                    model.foldingnet.folding_seed = saved_seed
                distances, ratios = [], []
                rows = json.loads((dataset/f'{split}.json').read_text())
                for row in rows:
                    xyz = torch.from_numpy(np.load(dataset/row['partial'])).unsqueeze(0).cuda()
                    truth = torch.from_numpy(np.load(dataset/row['complete'])).unsqueeze(0).cuda()
                    prediction = model(xyz)[-1]
                    if not torch.isfinite(prediction).all():
                        raise ValueError('Non-finite forward output')
                    distances.append(float(criterion(prediction, truth))*row['normalization']['radius_m']*1000)
                    patch = prediction[0, :report['generated_points']].cpu().numpy().reshape(report['queries'], -1, 3)
                    singular = np.linalg.svd(patch-patch.mean(1, keepdims=True), compute_uv=False)
                    ratios.extend((singular[:, 1]/np.maximum(singular[:, 0], 1e-10)).tolist())
                result[f'{variant}_{split}'] = dict(samples=len(rows), mean_chamfer_l1_mm=float(np.mean(distances)),
                    line_like_patch_fraction=float((np.asarray(ratios)<.1).mean()))
    finally:
        model.foldingnet.folding_seed = saved_seed
    result['note'] = 'Random seed is an inference-only probe, not a retrained model or proposed deployment fix.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, nargs='+', default=[
        ROOT/'experiments/train/MuJoCo_models/mujoco_v5', ROOT/'experiments/train/MuJoCo_models/mujoco_v6'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu', action='store_true', help='Also forward train/val/test and probe random decoder seeds')
    args = parser.parse_args()
    output = args.output or args.experiments[-1]/f'diagnosis_{time.time_ns()}'
    output.mkdir(parents=True, exist_ok=False)
    reports = [audit_experiment(p.resolve()) for p in args.experiments]
    if args.gpu:
        for report in reports:
            report['gpu_forward_audit'] = gpu_audit(report)
    (output/'report.json').write_text(json.dumps(reports, indent=2))
    for report in reports:
        plot_patches(report, output/f'{Path(report["experiment"]).name}_patches.png')
        print(json.dumps({key: value for key, value in report.items() if key != 'per_sample'}, indent=2))
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
