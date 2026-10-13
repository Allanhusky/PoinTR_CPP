"""Build a paired PoinTr dataset from collected MuJoCo view directories."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def points(path):
    value = np.load(path, allow_pickle=False).astype(np.float32)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError(f'Invalid XYZ array: {path.name}')
    return value


def resample(value, count, rng):
    if len(value) >= count:
        return value[rng.choice(len(value), count, replace=False)]
    indices = np.concatenate([np.arange(len(value)), rng.choice(len(value), count-len(value), replace=True)])
    return value[indices]


def matches_profile(metadata, cfg):
    if not np.isclose(metadata.get('object_rotation_x_deg', 0), cfg.get('object_rotation_x_deg', 0)):
        return False
    if not np.isclose(metadata['mesh_scale'], cfg['mesh_scale']):
        return False
    if Path(metadata['source_mesh']).name != Path(cfg['mesh']).name:
        return False
    desired = cfg.get('density', {'mode': 'native_pixels'})
    actual = metadata.get('density', {'mode': 'native_pixels'})
    if actual.get('mode') != desired.get('mode'):
        return False
    if actual.get('sampling', 'regular_grid') != desired.get('sampling', 'regular_grid'):
        return False
    if metadata.get('sensor_model', {}).get('enabled', False) != cfg.get('sensor_model', {}).get('enabled', False):
        return False
    if desired['mode'] == 'reference_spacing':
        if any(not np.isclose(actual.get(k, -1), desired[k]) for k in
               ['reference_depth_m', 'spacing_x_m', 'spacing_y_m']):
            return False
    return all(np.isclose(metadata['camera'][k], cfg['camera'][k]) for k in
               ['width', 'height', 'fx', 'fy', 'cx', 'cy'])


def split_groups(samples, seed, train_ratio, val_ratio):
    groups = sorted(set(sample['group'] for sample in samples))
    if len(groups) < 3:
        raise ValueError('至少需要 3 个方位角区间才能生成不重叠的 train/val/test；请采集更多角度或调小 --sector-deg')
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    nval = max(1, round(len(groups)*val_ratio))
    ntest = max(1, round(len(groups)*(1-train_ratio-val_ratio)))
    while nval+ntest >= len(groups):
        if nval > ntest:
            nval -= 1
        else:
            ntest -= 1
    assignment = {group: 'val' if i < nval else 'test' if i < nval+ntest else 'train'
                  for i, group in enumerate(groups)}
    return {name: [s for s in samples if assignment[s['group']] == name]
            for name in ['train', 'val', 'test']}


def build(args):
    source = args.source.resolve()
    if not source.is_dir():
        raise ValueError(f'Source directory does not exist: {source}')
    cfg = json.loads(args.sim_config.read_text())
    samples, rejected, seen = [], [], set()
    for metadata_path in sorted(source.rglob('metadata.json')):
        view = metadata_path.parent
        relative = view.relative_to(source)
        if not args.include_demos and any(p == 'demo' or p.endswith('_demo') for p in relative.parts):
            rejected.append(dict(source=str(relative), reason='demo directory'))
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
            if not args.all_profiles and not matches_profile(metadata, cfg):
                rejected.append(dict(source=str(relative), reason='different acquisition profile'))
                continue
            obj = points(view / 'partial_object.npy')
            gt = points(view / 'complete_object.npy')
            if len(obj) < args.min_raw_points:
                raise ValueError(f'Too few raw points: {len(obj)} < {args.min_raw_points}')
            rotation = np.asarray(metadata.get('R_source_object', np.eye(3)), dtype=float)
            camera_transform = np.asarray(metadata['T_world_camera_optical'], dtype=float)
            center = np.asarray(metadata['object_center_world_m'], dtype=float)
            if rotation.shape != (3, 3) or camera_transform.shape != (4, 4) or center.shape != (3,):
                raise ValueError('Invalid coordinate transform shape')
            if not np.isfinite(rotation).all() or not np.isfinite(camera_transform).all() or not np.isfinite(center).all():
                raise ValueError('Non-finite coordinate transforms')
            # Camera conversion uses the same rigid transform for input and truth.
            if args.frame == 'camera':
                obj = ((obj+center-camera_transform[:3, 3]) @ camera_transform[:3, :3]).astype(np.float32)
                gt = ((gt+center-camera_transform[:3, 3]) @ camera_transform[:3, :3]).astype(np.float32)
            digest = hashlib.sha256(obj.tobytes()).hexdigest()
            if digest in seen:
                rejected.append(dict(source=str(relative), reason='duplicate raw partial point cloud'))
                continue
            sample_seed = int(hashlib.sha256((str(args.seed)+str(relative)).encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(sample_seed)
            partial = resample(obj, args.input_points, rng)
            complete = resample(gt, args.gt_points, rng)
            centroid = partial.mean(axis=0)
            radius = float(np.linalg.norm(partial-centroid, axis=1).max())
            if not np.isfinite(radius) or radius < 1e-6:
                raise ValueError('Degenerate partial point cloud')
            azimuth = float(metadata['pose']['azimuth_deg']) % 360
            if not np.isfinite(azimuth):
                raise ValueError('Invalid azimuth')
            identity = hashlib.sha256(str(relative).encode()).hexdigest()[:16]
            group = int(np.floor(azimuth/args.sector_deg))
            samples.append(dict(id=identity, taxonomy_id='mujoco_object', group=group,
                                source=str(view), pose=metadata['pose'],
                                partial=f'partial/{identity}.npy', complete=f'complete/{identity}.npy',
                                raw_points=len(obj), unique_input_points=len(np.unique(partial, axis=0)),
                                normalization=dict(centroid=centroid.tolist(), radius_m=radius, frame=args.frame),
                                source_metadata=metadata,
                                arrays=((partial-centroid)/radius, (complete-centroid)/radius)))
            seen.add(digest)
        except (ValueError, KeyError, OSError, TypeError) as error:
            rejected.append(dict(source=str(relative), reason=str(error)))
    if not samples:
        raise ValueError(f'No usable views. Rejected: {json.dumps(rejected[:5], ensure_ascii=False)}')
    splits = split_groups(samples, args.seed, args.train_ratio, args.val_ratio)
    output = (args.output or ROOT / 'data' / 'MuJoCoObject' / f'dataset_{time.time_ns()}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'partial').mkdir()
    (output / 'complete').mkdir()
    (output / 'metadata').mkdir()
    for sample in samples:
        partial, complete = sample.pop('arrays')
        np.save(output / sample['partial'], partial.astype(np.float32))
        np.save(output / sample['complete'], complete.astype(np.float32))
        (output / 'metadata' / f'{sample["id"]}.json').write_text(json.dumps(sample, indent=2))
    for name, rows in splits.items():
        (output / f'{name}.json').write_text(json.dumps(rows, indent=2))
    dataset_config = dict(NAME='MuJoCo', DATA_PATH=str(output), N_POINTS=args.gt_points,
                          INPUT_POINTS=args.input_points, CARS=False)
    base = output / 'loader.yaml'
    base.write_text(yaml.safe_dump(dataset_config, sort_keys=False))
    report = dict(source=str(source), output=str(output), frame=args.frame, seed=args.seed,
                  input_points=args.input_points, gt_points=args.gt_points,
                  sector_deg=args.sector_deg,
                  split_policy='entire azimuth sectors stay in one split; shared object across splits',
                  counts={name: len(rows) for name, rows in splits.items()},
                  sectors={name: sorted(set(s['group'] for s in rows)) for name, rows in splits.items()},
                  retained=len(samples), rejected=rejected, all_profiles=args.all_profiles)
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'README.md').write_text(
        f'# Generated paired dataset\n\nSet `dataset_path: {output}` in '
        '`cfgs/MuJoCo_models/train.yaml` to train on this dataset.\n'
        'loader.yaml is generated metadata; do not edit it manually.\n')
    print(json.dumps({k: report[k] for k in ['output', 'retained', 'counts', 'sectors']}, indent=2))
    print(f'Skipped: {len(rejected)}; details in {output / "report.json"}')
    print(f'Set dataset_path: {output} in cfgs/MuJoCo_models/train.yaml')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'cfgs/MuJoCo_models/dataset.yaml',
                        help='Dataset preparation YAML; CLI overrides YAML')
    parser.add_argument('--source', type=Path, default=ROOT / 'mojoco_sim/outputs')
    parser.add_argument('--output', type=Path, help='New directory; existing data will not be overwritten')
    parser.add_argument('--sim-config', type=Path, default=ROOT / 'mojoco_sim/config.json')
    parser.add_argument('--frame', choices=['object', 'camera'], default='object')
    parser.add_argument('--all-profiles', action='store_true', help='Allow old orientations/density profiles')
    parser.add_argument('--include-demos', action='store_true')
    parser.add_argument('--input-points', type=int, default=2048)
    parser.add_argument('--gt-points', type=int, default=8192)
    parser.add_argument('--min-raw-points', type=int, default=128)
    parser.add_argument('--sector-deg', type=float, default=30)
    parser.add_argument('--train-ratio', type=float, default=.7)
    parser.add_argument('--val-ratio', type=float, default=.15)
    parser.add_argument('--seed', type=int, default=42)
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        try:
            config = yaml.safe_load(preliminary.config.read_text())
            if not isinstance(config, dict):
                raise ValueError('config must be a mapping')
            defaults = config.copy()
            allowed = {'source', 'output', 'sim_config', 'frame', 'all_profiles', 'include_demos',
                       'input_points', 'gt_points', 'min_raw_points', 'sector_deg',
                       'train_ratio', 'val_ratio', 'seed'}
            unknown = defaults.keys() - allowed
            if unknown:
                raise ValueError(f'Unknown dataset configuration keys: {sorted(unknown)}')
            for key in ['source', 'output', 'sim_config']:
                if key in defaults and defaults[key] is not None:
                    value = Path(defaults[key])
                    defaults[key] = value if value.is_absolute() else ROOT / value
            parser.set_defaults(**defaults)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
            parser.error(f'Invalid YAML configuration: {error}')
    args = parser.parse_args()
    if args.frame not in ['object', 'camera']:
        parser.error('frame must be object or camera')
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio+args.val_ratio < 1):
        parser.error('train/val ratios must be positive and sum to less than one')
    if not 0 < args.sector_deg <= 120:
        parser.error('sector-deg must be in (0, 120]')
    if min(args.input_points, args.gt_points, args.min_raw_points) < 1:
        parser.error('point counts must be positive')
    if args.input_points < 512:
        parser.error('PoinTr grouper requires at least 512 input points')
    try:
        build(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Dataset generation failed: {error}\n')


if __name__ == '__main__':
    main()
