"""Verify imports, CUDA extensions and one PoinTr optimizer step."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, help='MuJoCo view directory; otherwise use synthetic points')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch
    from pointnet2_ops import pointnet2_utils
    from utils.config import cfg_from_yaml_file
    from tools import builder
    from extensions.emd.emd_module import emdModule

    for name in ['open3d', 'timm', 'cv2', 'tensorboardX', 'chamfer', 'emd',
                 'cubic_feature_sampling', 'gridding', 'gridding_distance']:
        importlib.import_module(name)
        print(f'Import OK: {name}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on the GPU host outside a GPU-restricted sandbox')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = cfg_from_yaml_file(str(ROOT / 'cfgs/ShapeNet55_models/PoinTr.yaml'))
    model = builder.model_builder(config.model).cuda().train()
    if args.sample:
        points = np.load(args.sample / 'input_normalized.npy').astype(np.float32)
        complete = np.load(args.sample / 'complete_normalized.npy').astype(np.float32)
        if points.shape != (2048, 3):
            raise ValueError('Expected 2048 input points')
        selected = np.random.default_rng(42).choice(len(complete), min(8192, len(complete)), replace=False)
        xyz = torch.from_numpy(points).unsqueeze(0).cuda()
        gt = torch.from_numpy(complete[selected]).unsqueeze(0).cuda()
    else:
        xyz = torch.rand(1, 2048, 3, device='cuda')-.5
        gt = torch.rand(1, 8192, 3, device='cuda')-.5
    indices = pointnet2_utils.furthest_point_sample(xyz.contiguous(), 128)
    assert indices.shape == (1, 128)
    gathered = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), indices)
    assert gathered.shape == (1, 3, 128)
    # Exercise the otherwise optional EMD CUDA path on equal 1024-point clouds.
    emd_input = xyz[:, :1024].contiguous()
    emd_input = ((emd_input-emd_input.amin()) /
                 (emd_input.amax()-emd_input.amin()).clamp_min(1e-8)).contiguous()
    emd_dist, _ = emdModule()(emd_input, emd_input.clone(), .005, 5)
    assert torch.isfinite(emd_dist).all()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    output = model(xyz)
    assert output[0].shape == (1, 192, 3)
    assert output[-1].shape == (1, 8192, 3)
    coarse, dense = model.get_loss(output, gt)
    loss = coarse+dense
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    parameter = model.reduce_map.weight
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameter.detach()), 'Optimizer did not update parameters'
    torch.cuda.synchronize()
    report = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
                  parameters=sum(p.numel() for p in model.parameters()),
                  input_shape=list(xyz.shape), output_shapes=[list(p.shape) for p in output],
                  loss_coarse=float(coarse.detach()), loss_dense=float(dense.detach()),
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2,
                  sample=str(args.sample) if args.sample else 'synthetic',
                  status='PASS: imports, FPS/gather, EMD, forward, loss, backward, AdamW step')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
