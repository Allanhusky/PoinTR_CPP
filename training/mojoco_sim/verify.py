"""Compare rendered depth against analytical intersections with the source STL."""
import tempfile
from pathlib import Path

import numpy as np


def ray_depth(triangles, origin, directions):
    nearest = np.full(len(directions), np.inf)
    for triangle in triangles:
        edge1, edge2 = triangle[1]-triangle[0], triangle[2]-triangle[0]
        h = np.cross(directions, edge2)
        det = h @ edge1
        valid = np.abs(det) > 1e-12
        inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
        s = origin-triangle[0]
        u = (h @ s)*inv
        q = np.cross(s, edge1)
        v = (directions @ q)*inv
        t = np.dot(edge2, q)*inv
        hit = valid & (u >= -1e-7) & (v >= -1e-7) & (u+v <= 1+1e-7) & (t > 0)
        nearest = np.minimum(nearest, np.where(hit, t, np.inf))
    return nearest


def verify(sim):
    c = sim.camera
    if sim.density.get('mode') == 'reference_spacing' and sim.density.get('sampling') == 'random_pixels':
        d = sim.density
        probability = d['reference_depth_m']**2 / (c['fx']*c['fy']*d['spacing_x_m']*d['spacing_y_m'])
        np.testing.assert_allclose(sim.density_mask.mean(), probability, rtol=.02)
        print(f'Random pixel sampling: fraction={sim.density_mask.mean():.4f}; expected={probability:.4f}')
    elif sim.density.get('mode') == 'reference_spacing':
        # A fronto-parallel synthetic plane checks physical spacing and 1/Z^2 density.
        rows = np.flatnonzero(sim.density_mask.any(axis=1))
        cols = np.flatnonzero(sim.density_mask.any(axis=0))
        d = sim.density
        reference = d['reference_depth_m']
        for z in [.5, 1., 1.5, 2.]:
            measured_x = np.mean(np.diff(cols))*z/c['fx']
            measured_y = np.mean(np.diff(rows))*z/c['fy']
            np.testing.assert_allclose([measured_x, measured_y],
                                       [d['spacing_x_m']*z/reference, d['spacing_y_m']*z/reference], rtol=.005)
            density = 1/(measured_x*measured_y)
            print(f'Frontal plane Z={z} m: spacing={measured_x*1000:.3f} x {measured_y*1000:.3f} mm, density={density:.0f} points/m^2')
        native_count = c['width']*c['height']
        assert 0 < sim.density_mask.sum() < native_count
        print(f'Density lattice: {sim.density_mask.sum()}/{native_count} native pixels')
    for pose in [(0, 0, .6), (45, 25, .6), (135, 45, .8), (270, 15, .4)]:
        sim.set_pose(*pose)
        rgb, depth, mask = sim.render()
        assert rgb.shape == (c['height'], c['width'], 3)
        assert depth.shape == (c['height'], c['width'])
        assert mask.sum() > 100, f'No target pixels at {pose}'
        v, u = np.nonzero(mask)
        indices = np.linspace(0, len(u)-1, min(2048, len(u)), dtype=int)
        u, v = u[indices], v[indices]
        rays = np.column_stack([(u-c['cx'])/c['fx'], (v-c['cy'])/c['fy'], np.ones(len(u))])
        expected = ray_depth(sim.triangles+sim.center, sim.T_world_camera[:3, 3],
                             rays @ sim.T_world_camera[:3, :3].T)
        finite = np.isfinite(expected)
        assert finite.mean() > .98, f'Projection/mesh alignment error: {finite.mean()}'
        error = np.abs(depth[v[finite], u[finite]]-expected[finite])
        assert np.quantile(error, .99) < 2e-5, f'Depth mismatch: {np.quantile(error, .99)} m'
        print(f'pose={pose}: pixels={mask.sum()}, depth error p99={np.quantile(error, .99):.9f} m')
    with tempfile.TemporaryDirectory() as folder:
        destination = sim.save(Path(folder) / 'view')
        points = np.load(destination / 'input_2048.npy')
        assert points.shape == (sim.cfg['capture']['input_points'], 3)
        assert points.dtype == np.float32
        assert np.isfinite(points).all()
        import json
        metadata = json.loads((destination / 'metadata.json').read_text())
        normalized = np.load(destination / 'input_normalized.npy')
        normalization = metadata['normalization']
        restored = normalized*normalization['radius_m'] + normalization['centroid_object_m']
        np.testing.assert_allclose(restored, points, atol=1e-7)
        measured = np.load(destination / 'measured_depth_m.npy')
        from PIL import Image
        mask = np.asarray(Image.open(destination / 'mask.png')) > 0
        partial = np.load(destination / 'partial_camera.npy')
        assert mask.sum() == len(partial)
        v, u = np.nonzero(mask)
        np.testing.assert_allclose(partial[:, 2], measured[v, u], atol=1e-7)
        np.testing.assert_allclose(partial[:, 0]/partial[:, 2]*c['fx']+c['cx'], u, atol=1e-4)
        np.testing.assert_allclose(partial[:, 1]/partial[:, 2]*c['fy']+c['cy'], v, atol=1e-4)
    print('PASS: calibrated projection, STL alignment, depth, point export and normalization')
