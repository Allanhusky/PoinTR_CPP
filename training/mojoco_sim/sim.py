"""Calibrated Gemini2 pinhole acquisition in a static MuJoCo scene."""
import argparse
import itertools
import json
import os
from pathlib import Path
import struct
import time
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def read_stl(path):
    raw = path.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError('需要二进制 STL 文件')
    records = np.frombuffer(raw, dtype=np.dtype([
        ('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')]), offset=84)
    return records.copy()


def sample_surface(triangles, count, rng):
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    selected = triangles[rng.choice(len(triangles), count, p=area / area.sum())]
    r = np.sqrt(rng.random((count, 1)))
    s = rng.random((count, 1))
    return ((1-r)*selected[:, 0] + r*(1-s)*selected[:, 1] + r*s*selected[:, 2]).astype('float32')


def sample_points(points, count, rng):
    if len(points) == 0:
        raise ValueError('当前视角没有有效目标点，请调整相机距离/角度或深度范围')
    if len(points) >= count:
        return points[rng.choice(len(points), count, replace=False)]
    return points[np.concatenate([np.arange(len(points)),
                                 rng.choice(len(points), count-len(points), replace=True)])]


def density_grid(camera, density, rng=None):
    """Thin native pixels regularly or randomly, at a requested area density."""
    height, width = camera['height'], camera['width']
    if density.get('mode', 'native_pixels') == 'native_pixels':
        return np.ones((height, width), dtype=bool)
    if density['mode'] != 'reference_spacing':
        raise ValueError('density.mode must be native_pixels or reference_spacing')
    reference = density['reference_depth_m']
    sx, sy = density['spacing_x_m'], density['spacing_y_m']
    if min(reference, sx, sy) <= 0 or not np.isfinite([reference, sx, sy]).all():
        raise ValueError('Density reference depth and spacing must be finite and positive')
    stride_x = camera['fx']*sx/reference
    stride_y = camera['fy']*sy/reference
    if min(stride_x, stride_y) < 1:
        raise ValueError('Requested density exceeds native pixel density; cannot create new measurements')
    sampling = density.get('sampling', 'regular_grid')
    if sampling == 'random_pixels':
        if rng is None:
            raise ValueError('random_pixels requires a seeded RNG')
        # 在真实像素射线上随机保留点；期望面积密度与原稀疏网格相同。
        return rng.random((height, width)) < 1/(stride_x*stride_y)
    if sampling != 'regular_grid':
        raise ValueError('density.sampling must be regular_grid or random_pixels')

    def indices(size, principal, stride):
        k = np.arange(int(np.ceil((-.5-principal)/stride)),
                      int(np.ceil((size-.5-principal)/stride)))
        pixels = np.rint(principal+k*stride).astype(int)
        return np.unique(pixels[(pixels >= 0) & (pixels < size)])

    cols = indices(width, camera['cx'], stride_x)
    rows = indices(height, camera['cy'], stride_y)
    grid = np.zeros((height, width), dtype=bool)
    grid[np.ix_(rows, cols)] = True
    return grid


def sensor_depth(depth, ideal_mask, sampling_mask, camera, capture, sensor, rng):
    """Approximate depth defects on native pixel rays, not arbitrary XYZ jitter."""
    z = depth.astype(np.float64).copy()
    valid = ideal_mask & sampling_mask
    enabled = sensor.get('enabled', False)
    if enabled:
        # 用目标轮廓及深度突变识别边缘；不跨物体/背景插值深度。
        edge = np.zeros_like(ideal_mask)
        for axis in [0, 1]:
            left = [slice(None), slice(None)]
            right = left.copy()
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            boundary = (ideal_mask[left] != ideal_mask[right]) | (
                ideal_mask[left] & ideal_mask[right] &
                (np.abs(depth[left]-depth[right]) > sensor['edge_depth_jump_m']))
            edge[left] |= boundary
            edge[right] |= boundary
        for _ in range(sensor['edge_band_px']-1):
            expanded = np.pad(edge, 1)
            edge = (expanded[1:-1, 1:-1] | expanded[:-2, 1:-1] |
                    expanded[2:, 1:-1] | expanded[1:-1, :-2] | expanded[1:-1, 2:])
        # 随 Z 增大而增大的噪声；参数是近似模型，不是厂商精度承诺。
        sigma = np.sqrt(sensor['noise_floor_m']**2 +
                        (sensor['noise_std_m_at_1m']*z**2)**2)
        sigma *= np.where(edge, sensor['edge_noise_multiplier'], 1.)
        fraction = sensor['correlated_fraction']
        noise = rng.normal(size=z.shape)
        if fraction > 0:
            h, w = z.shape
            scale = sensor['correlation_px']
            coarse = rng.normal(size=(max(2, int(np.ceil(h/scale))), max(2, int(np.ceil(w/scale)))))
            correlated = np.asarray(Image.fromarray(coarse.astype('float32')).resize(
                (w, h), Image.Resampling.BILINEAR)).astype(float)
            correlated = (correlated-correlated.mean())/max(correlated.std(), 1e-8)
            noise = np.sqrt(1-fraction)*noise + np.sqrt(fraction)*correlated
        z += noise*sigma
        # 局部小空洞；只以有效目标像素为中心，模拟局部匹配失败。
        v, u = np.nonzero(ideal_mask)
        radius = sensor['hole_radius_px']
        n_holes = int(round(len(v)*sensor['hole_fraction']/(np.pi*radius**2)))
        for index in rng.choice(len(v), n_holes, replace=True) if len(v) else []:
            cy, cx = v[index], u[index]
            y0, y1 = max(0, cy-radius), min(z.shape[0], cy+radius+1)
            x0, x1 = max(0, cx-radius), min(z.shape[1], cx+radius+1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            valid[y0:y1, x0:x1] &= (yy-cy)**2+(xx-cx)**2 > radius**2
        valid &= ~(edge & (rng.random(z.shape) < sensor['edge_dropout']))
        outliers = valid & (rng.random(z.shape) < sensor['outlier_fraction'])
        z[outliers] += rng.normal(0, sensor['outlier_std_m'], int(outliers.sum()))
        quantum = sensor['quantization_m']
        if quantum > 0:
            z = np.rint(z/quantum)*quantum
    # 保留原有 capture 噪声/丢点开关，可额外叠加；默认额外噪声为零。
    if capture['noise_std_m']:
        z += rng.normal(0, capture['noise_std_m'], z.shape)
    valid &= rng.random(z.shape) >= capture['dropout']
    valid &= np.isfinite(z) & (z >= camera['min_depth_m']) & (z <= camera['max_depth_m'])
    return np.where(valid, z, 0).astype('float32'), valid


def write_ply(path, points):
    with path.open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\n'
                'property float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f, points, fmt='%.7f')


class Simulation:
    def __init__(self, config_path):
        import mujoco
        self.mj = mujoco
        self.config_path = Path(config_path).resolve()
        self.cfg = json.loads(self.config_path.read_text())
        self.camera = self.cfg['camera']
        self.density = self.cfg.get('density', {'mode': 'native_pixels'})
        self.rng = np.random.default_rng(self.cfg['capture']['seed'])
        self.density_mask = density_grid(self.camera, self.density, self.rng)
        self.sensor = self.cfg.get('sensor_model', {'enabled': False})
        if self.sensor.get('enabled', False):
            for key in ['noise_floor_m', 'noise_std_m_at_1m', 'quantization_m', 'outlier_std_m']:
                if not np.isfinite(self.sensor[key]) or self.sensor[key] < 0:
                    raise ValueError(f'sensor_model.{key} must be finite and nonnegative')
            for key in ['correlated_fraction', 'edge_dropout', 'hole_fraction', 'outlier_fraction']:
                if not np.isfinite(self.sensor[key]) or not 0 <= self.sensor[key] <= 1:
                    raise ValueError(f'sensor_model.{key} must be in [0, 1]')
            for key in ['edge_band_px', 'hole_radius_px', 'correlation_px']:
                if not isinstance(self.sensor[key], int) or self.sensor[key] < 1:
                    raise ValueError(f'sensor_model.{key} must be a positive integer')
            for key in ['edge_noise_multiplier', 'edge_depth_jump_m']:
                if not np.isfinite(self.sensor[key]) or self.sensor[key] <= 0:
                    raise ValueError(f'sensor_model.{key} must be finite and positive')
        c = self.camera
        if min(c['fx'], c['fy'], c['width'], c['height']) <= 0:
            raise ValueError('焦距和分辨率必须大于 0')
        if not 0 < c['near_m'] < c['far_m'] or not 0 < c['min_depth_m'] < c['max_depth_m']:
            raise ValueError('深度和裁剪范围无效')
        cap = self.cfg['capture']
        if not 0 <= cap['dropout'] < 1 or cap['noise_std_m'] < 0 or cap['input_points'] < 1:
            raise ValueError('丢点比例、噪声或输入点数无效')
        self.mesh_path = (self.config_path.parent / self.cfg['mesh']).resolve()
        records = read_stl(self.mesh_path)
        vertices = records['vertices'].astype(float)
        self.raw_center = (vertices.min(axis=(0, 1)) + vertices.max(axis=(0, 1))) / 2
        scale = self.cfg['mesh_scale']
        if scale <= 0:
            raise ValueError('mesh_scale 必须大于 0')
        # Bake placement rotation into the centered mesh and exported object frame.
        # +90 degrees about X maps the source long axis (+Y) to world +Z.
        angle = np.deg2rad(self.cfg.get('object_rotation_x_deg', 0.0))
        if not np.isfinite(angle):
            raise ValueError('object_rotation_x_deg must be finite')
        co, si = np.cos(angle), np.sin(angle)
        self.R_source_object = np.array([[1., 0., 0.], [0., co, -si], [0., si, co]])
        self.triangles = ((vertices - self.raw_center) * scale) @ self.R_source_object.T
        self.center = np.array([0., 0., -self.triangles[:, :, 2].min() + self.cfg['ground_clearance_m']])
        records['vertices'] = self.triangles
        records['normal'] = records['normal'].astype(float) @ self.R_source_object.T
        asset = ROOT / 'assets' / 'object_centered.stl'
        asset.parent.mkdir(exist_ok=True)
        asset.write_bytes(b'Centered metric visual mesh'.ljust(80, b' ') +
                          struct.pack('<I', len(records)) + records.tobytes())
        xml = f'''<mujoco model="Gemini2 object acquisition">
  <compiler angle="degree"/>
  <option gravity="0 0 0"/>
  <statistic center="0 0 {self.center[2]}" extent="1"/>
  <visual><global offwidth="{c['width']}" offheight="{c['height']}"/>
    <quality offsamples="0"/><map znear="{c['near_m']}" zfar="{c['far_m']}"/></visual>
  <asset><mesh name="object_mesh" file="{escape(str(asset))}"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".18 .20 .24" rgb2=".28 .30 .34" width="128" height="128"/>
    <material name="floor" texture="grid" texrepeat="10 10"/></asset>
  <worldbody>
    <light pos="0 -1 2" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="ground" type="plane" size="2 2 .01" material="floor"/>
    <body name="object" pos="0 0 {self.center[2]}">
      <geom name="target" type="mesh" mesh="object_mesh" rgba=".2 .65 .85 1" contype="0" conaffinity="0"/>
    </body>
    <camera name="gemini2" pos=".6 0 .3" resolution="{c['width']} {c['height']}"
      sensorsize=".0064 .0048" focalpixel="{c['fx']} {c['fy']}" ipd="0"/>
  </worldbody>
</mujoco>'''
        (ROOT / 'scene.xml').write_text(xml)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'gemini2')
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'target')
        self.renderer = mujoco.Renderer(self.model, height=c['height'], width=c['width'])
        self.set_pose(cap['azimuth_deg'], cap['elevation_deg'], cap['distance_m'])

    def set_pose(self, azimuth, elevation, distance):
        if distance <= 0 or not -89 <= elevation <= 89:
            raise ValueError('距离必须为正，仰角必须在 [-89, 89] 度范围内')
        self.pose = dict(azimuth_deg=float(azimuth), elevation_deg=float(elevation), distance_m=float(distance))
        a, e = np.deg2rad([azimuth, elevation])
        position = self.center + distance * np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
        z = (position - self.center) / distance  # MuJoCo camera looks along -Z.
        x = np.cross([0., 0., 1.], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rotation = np.column_stack([x, y, z])
        quat = np.empty(4)
        self.mj.mju_mat2Quat(quat, rotation.ravel())
        self.model.cam_pos[self.camera_id] = position
        self.model.cam_quat[self.camera_id] = quat
        self.mj.mj_forward(self.model, self.data)
        # Optical frame: right, down, forward. Row vectors transform with R.T.
        self.T_world_camera = np.eye(4)
        self.T_world_camera[:3, :3] = rotation @ np.diag([1., -1., -1.])
        self.T_world_camera[:3, 3] = position

    def update_scene(self):
        self.renderer.update_scene(self.data, camera='gemini2')
        c = self.camera
        # Explicit asymmetric frustum, including pixel-center convention.
        # Avoid relying on fovy or camera principalpixel sign conventions.
        for cam in self.renderer.scene.camera:
            n = c['near_m']
            cam.frustum_near = n
            cam.frustum_far = c['far_m']
            cam.frustum_width = n * c['width'] / (2*c['fx'])
            cam.frustum_center = n * ((c['width']-1)/2-c['cx']) / c['fx']
            cam.frustum_top = n * (c['cy']+.5) / c['fy']
            cam.frustum_bottom = n * (c['cy']+.5-c['height']) / c['fy']

    def render(self):
        self.update_scene()
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        c = self.camera
        self.ideal_mask = ((segmentation[..., 0] == self.target_id) &
                (segmentation[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM)) &
                np.isfinite(depth) & (depth >= c['min_depth_m']) & (depth <= c['max_depth_m']))
        if self.density.get('sampling') == 'random_pixels':
            self.density_mask = density_grid(self.camera, self.density, self.rng)
        mask = self.ideal_mask & self.density_mask
        return rgb, depth, mask

    def point_cloud(self, depth, mask):
        cap, c = self.cfg['capture'], self.camera
        self.measured_depth, self.measured_mask = sensor_depth(
            depth, self.ideal_mask, mask, c, cap, self.sensor, self.rng)
        v, u = np.nonzero(self.measured_mask)
        z = self.measured_depth[v, u].astype(float)
        camera = np.column_stack([(u-c['cx'])*z/c['fx'], (v-c['cy'])*z/c['fy'], z])
        world = camera @ self.T_world_camera[:3, :3].T + self.T_world_camera[:3, 3]
        return camera.astype('float32'), world.astype('float32'), (world-self.center).astype('float32')

    def save(self, destination, images=None):
        rgb, depth, mask = images if images is not None else self.render()
        camera, world, obj = self.point_cloud(depth, mask)
        sampled = sample_points(obj, self.cfg['capture']['input_points'], self.rng)
        # Derived only from visible input; invert using saved centroid and radius.
        centroid = sampled.mean(axis=0)
        radius = float(np.linalg.norm(sampled-centroid, axis=1).max())
        if radius <= 1e-8:
            raise ValueError('点云范围过小，无法归一化')
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        for name, points in [('partial_camera', camera), ('partial_world', world),
                             ('partial_object', obj), ('input_2048', sampled),
                             ('input_normalized', (sampled-centroid)/radius)]:
            np.save(destination / f'{name}.npy', points.astype('float32'))
        gt = sample_surface(self.triangles, self.cfg['capture']['ground_truth_points'], self.rng)
        np.save(destination / 'complete_object.npy', gt)
        np.save(destination / 'complete_normalized.npy', (gt-centroid)/radius)
        np.save(destination / 'depth_m.npy', depth)
        np.save(destination / 'measured_depth_m.npy', self.measured_depth)
        Image.fromarray(rgb).save(destination / 'rgb.png')
        Image.fromarray(mask.astype('uint8')*255).save(destination / 'sampling_mask.png')
        Image.fromarray(self.measured_mask.astype('uint8')*255).save(destination / 'mask.png')
        Image.fromarray(self.ideal_mask.astype('uint8')*255).save(destination / 'ideal_mask.png')
        Image.fromarray(np.clip(self.measured_depth*1000, 0, 65535).astype('uint16')).save(destination / 'target_depth_mm.png')
        write_ply(destination / 'partial_object.ply', obj)
        write_ply(destination / 'complete_object.ply', gt)
        metadata = dict(camera=camera_intrinsics(self.camera), pose=self.pose,
                        T_world_camera_optical=self.T_world_camera.tolist(),
                        object_center_world_m=self.center.tolist(),
                        source_mesh=str(self.mesh_path), source_center=self.raw_center.tolist(),
                        object_rotation_x_deg=self.cfg.get('object_rotation_x_deg', 0.0),
                        R_source_object=self.R_source_object.tolist(),
                        mesh_scale=self.cfg['mesh_scale'], raw_points=len(obj),
                        normalization=dict(centroid_object_m=centroid.tolist(), radius_m=radius),
                        noise_std_m=self.cfg['capture']['noise_std_m'], dropout=self.cfg['capture']['dropout'],
                        sensor_model=self.sensor,
                        density=dict(self.density,
                                     full_frame_sampling_pixels=int(self.density_mask.sum()),
                                     ideal_target_pixels=int(self.ideal_mask.sum()),
                                     sampled_target_pixels=int(mask.sum()),
                                     measured_target_pixels=int(self.measured_mask.sum()),
                                     spacing_depth_definition='optical Z; frontal plane',
                                     pixel_quantization='native pixel centers; random sampling preserves expected area density' if self.density.get('sampling') == 'random_pixels' else 'nearest native pixel; spacing is approximate'),
                        seed=self.cfg['capture']['seed'])
        (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print(f'Saved {destination} ({len(obj)} visible points)', flush=True)
        return destination

    def close(self):
        self.renderer.close()


def camera_intrinsics(c):
    return dict(c, K=[[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]],
                optical_axes='x right, y down, z forward', unit='meter',
                pixel_convention='integer coordinates are pixel centers',
                distortion='none; ideal pinhole')


def gui(sim, output):
    import tkinter as tk
    from tkinter import messagebox
    from PIL import ImageTk
    root = tk.Tk()
    root.title('MuJoCo · Gemini2 · Object point cloud capture')
    panel = tk.Frame(root)
    panel.pack()
    rgb_label, depth_label = tk.Label(panel), tk.Label(panel)
    rgb_label.grid(row=0, column=0)
    depth_label.grid(row=0, column=1)
    status = tk.StringVar()
    tk.Label(root, textvariable=status).pack()
    controls = []
    current = None
    dirty = True

    def changed(_=None):
        nonlocal dirty
        dirty = True

    for label, key, lo, hi, resolution in [
        ('Azimuth (deg)', 'azimuth_deg', 0, 360, 1),
        ('Elevation (deg)', 'elevation_deg', -20, 85, 1),
        ('Distance (m)', 'distance_m', .2, 2.5, .01)]:
        slider = tk.Scale(root, label=label, from_=lo, to=hi, resolution=resolution,
                          orient='horizontal', length=800, command=changed)
        slider.set(sim.pose[key])
        slider.pack()
        controls.append(slider)

    def refresh():
        nonlocal dirty, current
        if dirty:
            sim.set_pose(*(slider.get() for slider in controls))
            current = sim.render()
            rgb, depth, mask = current
            # 界面显示与导出使用相同传感器模型；每次采集重新生成随机观测。
            sim.point_cloud(depth, mask)
            depth, mask = sim.measured_depth, sim.measured_mask
            c = sim.camera
            intensity = np.clip(1-(depth-c['min_depth_m'])/(c['max_depth_m']-c['min_depth_m']), 0, 1)
            display = np.zeros_like(rgb)
            display[..., 1] = np.where(mask, intensity*255, 0).astype('uint8')
            display[..., 2] = np.where(mask, 255-intensity*160, 0).astype('uint8')
            for label, array in [(rgb_label, rgb), (depth_label, display)]:
                photo = ImageTk.PhotoImage(Image.fromarray(array))
                label.configure(image=photo)
                label.image = photo
            status.set(f'RGB | target depth     Visible points: {mask.sum()}     Units: meters')
            dirty = False
        root.after(60, refresh)

    def save():
        try:
            sim.set_pose(*(slider.get() for slider in controls))
            refresh_now = sim.render()
            destination = sim.save(output / f'view_{time.time_ns()}', refresh_now)
            status.set(f'Saved: {destination}')
        except Exception as error:
            messagebox.showerror('Capture failed', str(error))

    tk.Button(root, text='Save point cloud (S)', command=save).pack(pady=8)
    root.bind('<s>', lambda _: save())
    root.bind('<Escape>', lambda _: root.destroy())
    refresh()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['gui', 'capture', 'batch', 'check', 'viewer'])
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--azimuth', type=float)
    parser.add_argument('--elevation', type=float)
    parser.add_argument('--distance', type=float)
    parser.add_argument('--backend', choices=['egl', 'glfw', 'osmesa'], default=os.environ.get('MUJOCO_GL', 'egl'))
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = args.backend  # Must precede importing mujoco.
    sim = Simulation(args.config)
    try:
        sim.set_pose(args.azimuth if args.azimuth is not None else sim.pose['azimuth_deg'],
                     args.elevation if args.elevation is not None else sim.pose['elevation_deg'],
                     args.distance if args.distance is not None else sim.pose['distance_m'])
        if args.mode == 'gui':
            gui(sim, args.output)
        elif args.mode == 'viewer':
            import mujoco.viewer
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                viewer.cam.lookat[:] = sim.center
                viewer.cam.distance = .6
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.03)
        elif args.mode == 'capture':
            sim.save(args.output / f'view_{time.time_ns()}')
        elif args.mode == 'batch':
            # 每次采集新建目录；原有采集数据不会覆盖。
            run = args.output / f'batch_{time.time_ns()}'
            b = sim.cfg['batch']
            # 遍历方位角、仰角和距离的全部组合；数量等于三个列表长度的乘积。
            poses = list(itertools.product(b['azimuths_deg'], b['elevations_deg'], b['distances_m']))
            manifest = []
            run.mkdir(parents=True)
            try:
                for i, pose in enumerate(poses):
                    sim.set_pose(*pose)
                    try:
                        dest = sim.save(run / f'{i:04d}')
                        manifest.append(dict(view=dest.name, pose=sim.pose.copy(), status='saved'))
                    except ValueError as error:
                        # 当前视角无法采集时记录原因，继续采集其余视角。
                        manifest.append(dict(view=f'{i:04d}', pose=sim.pose.copy(), status='skipped', reason=str(error)))
                        print(f'Skipped {i}: {error}', flush=True)
            finally:
                # 即使采集中断，也保存已完成视角的状态清单。
                (run / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            print(f'Batch: {run}; saved {sum(x["status"] == "saved" for x in manifest)}/{len(poses)}')
        else:
            from verify import verify
            verify(sim)
    finally:
        sim.close()


if __name__ == '__main__':
    main()
