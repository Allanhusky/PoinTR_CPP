"""Evaluate a trained PoinTr checkpoint and export paired 3D visualizations."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_ply(path, xyz):
    import numpy as np
    with path.open('w') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n'
                     'property float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(stream, xyz, fmt='%.8f')


def limits(clouds):
    import numpy as np
    combined = np.concatenate(clouds)
    center = (combined.min(0)+combined.max(0))/2
    half = max(float(np.ptp(combined, axis=0).max())*.55, 1.)
    return [(float(c-half), float(c+half)) for c in center]


def plot_row(fig, grid, index, sample):
    clouds, errors = sample['clouds'], sample['error_mm']
    bounds = limits(clouds)
    titles = ['Input', 'PoinTr prediction', 'Ground truth', 'Prediction error / GT overlay']
    for col in range(4):
        ax = fig.add_subplot(grid[index, col], projection='3d')
        if col < 3:
            xyz = clouds[col]
            ax.scatter(*xyz.T, s=.6, c=['#2980b9', '#16a085', '#e67e22'][col],
                       alpha=.7, linewidths=0, rasterized=True)
        else:
            ax.scatter(*clouds[2].T, s=.5, c='#bbbbbb', alpha=.2, linewidths=0)
            ax.scatter(*clouds[1].T, s=.8, c=errors, cmap='turbo', vmin=0, vmax=10,
                       alpha=.8, linewidths=0, rasterized=True)
        ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=40)
        ax.set_xlabel('X (mm)', fontsize=7)
        ax.set_ylabel('Y (mm)', fontsize=7)
        ax.set_zlabel('Z (mm)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(titles[col], fontsize=9)
    row = sample['metrics']
    position = grid[index, 0].get_position(fig)
    fig.text(.005, (position.y0+position.y1)/2, f'{row["id"][:8]}\naz {row["azimuth_deg"]:.0f} deg\n'
             f'dist {row["distance_m"]:.2f} m\n'
             f'CD {row["chamfer_l1_mm"]:.2f} mm\nF@2mm {row["fscore_2mm"]:.2f}', fontsize=8)


def export_visuals(samples, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    for sample in samples:
        fig = plt.figure(figsize=(14, 3.6))
        grid = fig.add_gridspec(1, 4, left=.1, right=.98)
        plot_row(fig, grid, 0, sample)
        fig.savefig(output / sample['metrics']['id'] / 'comparison.png', dpi=150)
        plt.close(fig)
    order = np.argsort([s['metrics']['chamfer_l1_mm'] for s in samples])
    selected = np.unique(np.linspace(0, len(order)-1, min(6, len(order))).round().astype(int))
    fig = plt.figure(figsize=(14, 4.1*len(selected)))
    grid = fig.add_gridspec(len(selected), 4, left=.1, right=.95, top=.97, bottom=.03, hspace=.4)
    for index, rank in enumerate(selected):
        plot_row(fig, grid, index, samples[order[rank]])
    bar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, 10), cmap='turbo'),
                      cax=fig.add_axes([.967, .15, .012, .7]))
    bar.set_label('Prediction to GT nearest distance (mm); clipped at 10')
    fig.savefig(output / 'overview.png', dpi=150)
    plt.close(fig)

    fig = make_subplots(rows=1, cols=4, specs=[[{'type': 'scene'}]*4],
                        subplot_titles=['Input', 'Prediction', 'Ground truth', 'Error / GT overlay'])
    for index, sample in enumerate(samples):
        for col in range(3):
            xyz = sample['clouds'][col]
            fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
                marker=dict(size=1.5, color=['#2980b9', '#16a085', '#e67e22'][col]),
                name=['Input', 'Prediction', 'Ground truth'][col], visible=index == 0), row=1, col=col+1)
        xyz = sample['clouds'][2]
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
            marker=dict(size=1, color='#aaaaaa', opacity=.25), name='GT overlay', visible=index == 0), row=1, col=4)
        xyz = sample['clouds'][1]
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
            marker=dict(size=1.5, color=sample['error_mm'], colorscale='Turbo', cmin=0, cmax=10,
                        colorbar=dict(title='mm', len=.65)),
            name='Prediction error', visible=index == 0), row=1, col=4)

    def layout(sample):
        row = sample['metrics']
        bounds = limits(sample['clouds'])
        result = dict(title=dict(text=f'Test {row["id"]} | az {row["azimuth_deg"]:.0f} deg | '
                      f'distance {row["distance_m"]:.2f} m | '
                      f'CD-L1 {row["chamfer_l1_mm"]:.2f} mm | F@2mm {row["fscore_2mm"]:.3f}'))
        for i in range(4):
            scene = 'scene' + (str(i+1) if i else '')
            result[scene] = dict(aspectmode='cube', camera=dict(eye=dict(x=1.5, y=1.5, z=1)),
                **{axis+'axis': dict(title=axis.upper()+' (mm)', range=bounds[j])
                   for j, axis in enumerate('xyz')})
        return result

    buttons = []
    for index, sample in enumerate(samples):
        buttons.append(dict(label=f'{index+1}: {sample["metrics"]["id"][:8]} | '
                            f'{sample["metrics"]["distance_m"]:.2f} m', method='update',
            args=[dict(visible=[i//5 == index for i in range(len(fig.data))]), layout(sample)]))
    fig.update_layout(**layout(samples[0]), height=650, showlegend=False,
        margin=dict(l=0, r=30, b=0, t=130),
        updatemenus=[dict(buttons=buttons, x=0, y=1.15, xanchor='left')])
    fig.write_html(output / 'interactive.html', include_plotlyjs=True, full_html=True)


DISTANCE_DEFINITION = 'Camera optical center to object bounding-box center, in meters; not optical Z depth.'


def distance_metadata(metrics):
    distances = sorted({float(row['distance_m']) for row in metrics})
    return dict(distance_definition=DISTANCE_DEFINITION, distances_m=distances,
                samples_by_distance_m={str(d): sum(float(row['distance_m']) == d for row in metrics)
                                       for d in distances})


def annotate_existing(output):
    """Add recorded distances and rebuild visuals from saved predictions on CPU."""
    import numpy as np
    from scipy.spatial import cKDTree

    output = output.resolve()
    with (output / 'metrics.csv').open() as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        metrics = list(reader)
    if not metrics:
        raise ValueError('Empty saved test results')
    summary = json.loads((output / 'summary.json').read_text())
    samples, metadata = [], []
    # Validate all saved samples before modifying result files.
    for row in metrics:
        folder = output / row['id']
        meta = json.loads((folder / 'metadata.json').read_text())
        if meta['sample']['id'] != row['id'] or meta['metrics']['id'] != row['id']:
            raise ValueError(f'Mismatched sample ID in {folder}')
        distance = float(meta['sample']['pose']['distance_m'])
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError(f'Invalid recorded distance in {folder}')
        row['distance_m'] = distance
        meta['metrics']['distance_m'] = distance
        clouds = [np.load(folder / f'{name}_m.npy', allow_pickle=False)
                  for name in ['input', 'prediction', 'ground_truth']]
        if any(c.ndim != 2 or c.shape[1] != 3 or not len(c) or not np.isfinite(c).all() for c in clouds):
            raise ValueError(f'Invalid saved point cloud in {folder}')
        # Preserve all original metric values; only recompute colors from saved clouds.
        plot_metrics = dict(meta['metrics'], distance_m=distance)
        samples.append(dict(metrics=plot_metrics, clouds=[c*1000 for c in clouds],
                            error_mm=cKDTree(clouds[2]).query(clouds[1])[0]*1000))
        metadata.append((folder / 'metadata.json', meta))
    if 'distance_m' not in fields:
        fields.insert(fields.index('azimuth_deg')+1, 'distance_m')
    for path, meta in metadata:
        path.write_text(json.dumps(meta, indent=2))
    with (output / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    summary.update(distance_metadata(metrics))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Added recorded distances to {len(samples)} samples. Rebuilding visuals...', flush=True)
    export_visuals(samples, output)
    print(f'Updated: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path,
        default=ROOT / 'experiments/PoinTr/MuJoCo_models/mujoco_object')
    parser.add_argument('--checkpoint', type=Path, help='Default: experiment/ckpt-best.pth')
    parser.add_argument('--config', type=Path, help='Default: experiment/config.yaml (saved training config)')
    parser.add_argument('--output', type=Path, help='New directory; defaults to experiment/test_vis_TIMESTAMP')
    parser.add_argument('--annotate-existing', type=Path,
                        help='Add distance labels to an existing test_vis directory without GPU inference')
    args = parser.parse_args()
    if args.annotate_existing is not None:
        annotate_existing(args.annotate_existing)
        return
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from tools import builder
    from utils.config import cfg_from_yaml_file
    from utils.metrics import Metrics

    checkpoint = (args.checkpoint or args.experiment / 'ckpt-best.pth').resolve()
    config_path = (args.config or args.experiment / 'config.yaml').resolve()
    config = cfg_from_yaml_file(str(config_path))
    if config.dataset.test._base_.NAME != 'MuJoCo':
        raise ValueError('This visualization requires the paired MuJoCo dataset')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the PoinTr custom operators')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dataset = Path(config.dataset.test._base_.DATA_PATH)
    rows = json.loads((dataset / 'test.json').read_text())
    if not rows:
        raise ValueError('Empty test set')
    output = (args.output or args.experiment / f'test_vis_{time.time_ns()}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = builder.model_builder(config.model).cuda().eval()
    builder.load_model(model, str(checkpoint))
    samples = []
    with torch.inference_mode():
        warmup = torch.from_numpy(np.load(dataset / rows[0]['partial'])).unsqueeze(0).cuda()
        for _ in range(10):
            model(warmup)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for row in rows:
            partial = np.load(dataset / row['partial']).astype(np.float32)
            truth = np.load(dataset / row['complete']).astype(np.float32)
            xyz = torch.from_numpy(partial).unsqueeze(0).cuda()
            gt = torch.from_numpy(truth).unsqueeze(0).cuda()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            predicted = model(xyz)[-1].contiguous()
            end.record()
            end.synchronize()
            elapsed = begin.elapsed_time(end)
            values = [float(value) for value in Metrics.get(predicted, gt, require_emd=False)[:3]]
            pred = predicted[0].cpu().numpy()
            if not np.isfinite(pred).all():
                raise ValueError(f'Non-finite prediction for {row["id"]}')
            norm = row['normalization']
            clouds_m = [cloud*norm['radius_m']+np.asarray(norm['centroid']) for cloud in [partial, pred, truth]]
            pred_dist = cKDTree(clouds_m[2]).query(clouds_m[1])[0]*1000
            gt_dist = cKDTree(clouds_m[1]).query(clouds_m[2])[0]*1000
            precision, recall = float((pred_dist < 2).mean()), float((gt_dist < 2).mean())
            result = dict(id=row['id'], azimuth_deg=row['pose']['azimuth_deg'],
                distance_m=float(row['pose']['distance_m']),
                frame=norm['frame'], input_points=len(partial), predicted_points=len(pred),
                fscore_normalized_001=values[0], CDL1_x1000=values[1], CDL2_x1000=values[2],
                chamfer_l1_mm=float((pred_dist.mean()+gt_dist.mean())/2),
                prediction_to_gt_mean_mm=float(pred_dist.mean()),
                prediction_to_gt_p95_mm=float(np.percentile(pred_dist, 95)),
                gt_to_prediction_mean_mm=float(gt_dist.mean()),
                fscore_2mm=2*precision*recall/(precision+recall) if precision+recall else 0.,
                precision_2mm=precision, recall_2mm=recall, gpu_forward_ms=elapsed)
            folder = output / row['id']
            folder.mkdir()
            np.save(folder / 'prediction_normalized.npy', pred)
            for name, cloud in zip(['input', 'prediction', 'ground_truth'], clouds_m):
                np.save(folder / f'{name}_m.npy', cloud.astype(np.float32))
                write_ply(folder / f'{name}_m.ply', cloud)
            (folder / 'metadata.json').write_text(json.dumps(dict(sample=row, metrics=result), indent=2))
            samples.append(dict(metrics=result, clouds=[c*1000 for c in clouds_m], error_mm=pred_dist))
            print(f'{len(samples)}/{len(rows)} {row["id"]}: distance={result["distance_m"]:.2f} m, '
                  f'CD={result["chamfer_l1_mm"]:.3f} mm', flush=True)
    metrics = [s['metrics'] for s in samples]
    summary = dict(checkpoint=str(checkpoint), config=str(config_path), dataset=str(dataset),
        split='test', samples=len(rows), gpu=torch.cuda.get_device_name(0),
        means={key: float(np.mean([r[key] for r in metrics])) for key in metrics[0]
               if key not in ['id', 'frame', 'azimuth_deg', 'distance_m', 'input_points', 'predicted_points']},
        peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
        notes=['CDL1/CDL2 use repository normalized metrics multiplied by 1000.',
               'Physical distances use each sample input-derived inverse normalization.',
               'F@2mm uses a 2 mm distance threshold; no EMD computed.',
               'GPU forward timing excludes preprocessing, transfers and visualization; host GPU, not Orin.',
               'Held-out azimuth sectors of the same object; not cross-object generalization.'])
    summary.update(distance_metadata(metrics))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (output / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print('Rendering PNG and offline interactive HTML...', flush=True)
    export_visuals(samples, output)
    print(json.dumps(summary, indent=2))
    print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
