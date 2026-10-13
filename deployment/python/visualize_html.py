"""Export saved native C++ predictions using the original Python review layout."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualize_cloud import load_cloud
ROOT=Path(__file__).resolve().parents[1]

def limits(clouds):
    import numpy as np
    combined = np.concatenate(clouds)
    center = (combined.min(0)+combined.max(0))/2
    half = max(float(np.ptp(combined, axis=0).max())*.55, 1.)
    return [(float(c-half), float(c+half)) for c in center]


def export_html(samples, output):
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
        result = dict(title=dict(text=f'C++ TensorRT {row["id"]} | az {row["azimuth_deg"]:.0f} deg | '
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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=(ROOT.parent/'training')/'data/MuJoCoObject/dataset_v8_far')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/html_review')
    p.add_argument('--completion',type=Path,default=ROOT/'outputs/completion.ply')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((a.dataset/'test.json').read_text())
    by_id={r['id']:r for r in rows}
    meta=json.loads((ROOT/'examples/metadata.json').read_text())
    row=next(r for r in rows if r['id']==meta['sample_id'])
    selected=[(row,load_cloud(ROOT/'examples/partial_camera_far.npy'),load_cloud(a.completion),'current_raw_input')]
    validation=json.loads((ROOT/'tests/results/validation.json').read_text())
    for case in validation['cases']:
        r=by_id[case['id']];n=r['normalization']
        restore=lambda cloud: cloud*n['radius_m']+np.asarray(n['centroid'])
        selected.append((r,restore(np.load(a.dataset/r['partial'])),restore(np.load(ROOT/'tests/results'/f'{r["id"]}.npy')),'normalized_test'))
    samples=[]
    for r,partial,pred,mode in selected:
        n=r['normalization']
        truth=np.load(a.dataset/r['complete'])*n['radius_m']+np.asarray(n['centroid'])
        clouds=[partial,pred,truth]
        if any(c.ndim!=2 or c.shape[1]!=3 or not np.isfinite(c).all() for c in clouds):
            raise ValueError('Invalid point cloud')
        pd=cKDTree(truth).query(pred)[0]*1000;gd=cKDTree(pred).query(truth)[0]*1000
        precision=float((pd<2).mean());recall=float((gd<2).mean())
        metrics=dict(id=r['id']+('_raw' if mode=='current_raw_input' else ''),
            azimuth_deg=r['pose']['azimuth_deg'],elevation_deg=r['pose']['elevation_deg'],
            distance_m=r['pose']['distance_m'],input_points=len(partial),predicted_points=len(pred),
            chamfer_l1_mm=float((pd.mean()+gd.mean())/2),
            fscore_2mm=2*precision*recall/(precision+recall) if precision+recall else 0.,
            precision_2mm=precision,recall_2mm=recall,input_mode=mode)
        samples.append(dict(metrics=metrics,clouds=[c*1000 for c in clouds],error_mm=pd))
    export_html(samples,a.output)
    metrics=[s['metrics'] for s in samples]
    (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    with (a.output/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    print(json.dumps(dict(html=str(a.output/'interactive.html'),samples=len(samples),current=metrics[0]),indent=2))
if __name__=='__main__':main()
