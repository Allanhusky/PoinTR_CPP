"""Render input and completed camera-coordinate point clouds to a PNG."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_cloud(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        return np.load(path)[:, :3]
    with path.open() as f:
        header=[]
        for line in f:
            header.append(line.strip())
            if line.strip() == 'end_header':
                break
        if 'format ascii 1.0' not in header:
            raise ValueError('Only ASCII PLY is supported')
        n=int(next(s.split()[-1] for s in header if s.startswith('element vertex ')))
        return np.asarray([list(map(float, next(f).split()[:3])) for _ in range(n)])

def main():
    p=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--input', type=Path, default=root/'examples/partial_camera_far.npy')
    p.add_argument('--completion', type=Path, default=root/'outputs/completion.ply')
    p.add_argument('--output', type=Path, default=root/'outputs/pointcloud_visualization.png')
    a=p.parse_args()
    partial=load_cloud(a.input);complete=load_cloud(a.completion)
    center=(complete.min(0)+complete.max(0))/2
    partial=(partial-center)*1000;complete=(complete-center)*1000
    extent=max(np.ptp(complete, axis=0).max(),np.ptp(partial, axis=0).max())*.56
    fig=plt.figure(figsize=(15,5),facecolor='#f7f9fc')
    for i,title in enumerate([f'Observed input ({len(partial)} points)',f'Completed output ({len(complete)} points)','Overlay: input / completion']):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        ax.set_facecolor('#f7f9fc')
        if i in (1,2):
            ax.scatter(*complete.T,s=1.4 if i==1 else 1,c='#3986cc',alpha=.65 if i==1 else .22,depthshade=False,rasterized=True)
        if i in (0,2):
            ax.scatter(*partial.T,s=9,c='#ec8236',alpha=.95,depthshade=False,rasterized=True)
        ax.set(xlim=(-extent,extent),ylim=(-extent,extent),zlim=(-extent,extent),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=title)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-58)
        ax.tick_params(labelsize=8)
    fig.suptitle('PoinTr TensorRT | Camera optical coordinates, centered for display',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=180,bbox_inches='tight');plt.close(fig)
    print(a.output)
if __name__=='__main__':main()
