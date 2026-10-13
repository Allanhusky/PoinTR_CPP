"""Benchmark saved training clouds with a persistent native TensorRT context."""
import argparse,csv,json,os,subprocess,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def stats(values):
    x=np.asarray(values)
    return dict(mean_ms=float(x.mean()),median_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)),p99_ms=float(np.percentile(x,99)),min_ms=float(x.min()),max_ms=float(x.max()),sequential_fps=float(1000/x.mean()))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=(ROOT.parent/'training')/'data/MuJoCoObject/dataset_v8_far')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/speed_benchmark')
    a=p.parse_args();a.dataset=a.dataset.resolve();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((a.dataset/'train.json').read_text())
    env=os.environ.copy();sdk=env.get('TENSORRT_ROOT',str(Path.home()/'opt/tensorrt/TensorRT-10.14.1.48'))
    env['LD_LIBRARY_PATH']=sdk+'/lib:'+str(ROOT/'build/lib')+':'+env.get('LD_LIBRARY_PATH','')
    report=dict(dataset=str(a.dataset),split='train',samples=len(rows),engine=str(ROOT/'models/pointr_fp32.engine'),precision='FP32',batch_size=1,warmup=50,modes={},notes=['Persistent model/context. Inputs preloaded into RAM before timing.','gpu_pipeline_ms includes H2D, inference, D2H. infer_ms also includes CPU preprocessing/inverse normalization.','Excludes disk IO, engine initialization, camera acquisition, YOLOE and ROS.','Meter-mode inputs are restored 2048-point training inputs, not original variable-size depth clouds.','Host measurement, not Orin performance.'])
    with tempfile.TemporaryDirectory(prefix='pointr_speed_') as temporary:
        temp=Path(temporary);normalized=[];meters=[]
        for i,r in enumerate(rows):
            path=a.dataset/r['partial'];x=np.load(path);n=r['normalization']
            assert x.shape==(2048,3) and np.isfinite(x).all()
            physical=(x*n['radius_m']+np.asarray(n['centroid'])).astype(np.float32)
            target=temp/f'{i}.npy';np.save(target,physical)
            normalized.append(str(path));meters.append(str(target))
        for mode,files in [('normalized',normalized),('meters',meters)]:
            manifest=temp/f'{mode}.txt';manifest.write_text('\n'.join(files)+'\n')
            target=a.output/f'{mode}.csv'
            run=subprocess.run([str(ROOT/'build/bin/pointr_benchmark'),str(ROOT/'models/pointr_fp32.engine'),str(manifest),str(target),mode],env=env,check=True,capture_output=True,text=True)
            timing=list(csv.DictReader(target.open()))
            result=json.loads(run.stdout);result.update(gpu_pipeline=stats([float(r['gpu_pipeline_ms']) for r in timing]),infer=stats([float(r['infer_ms']) for r in timing]))
            report['modes'][mode]=result;print(mode,json.dumps(result),flush=True)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print('Saved:',a.output/'summary.json')
if __name__=='__main__':main()
