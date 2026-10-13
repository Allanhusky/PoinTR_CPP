"""Compare native TensorRT C++ outputs against the original pretrained PyTorch model."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--source',type=Path,default=(ROOT.parent/'training'))
 p.add_argument('--engine',type=Path,default=ROOT/'models/pointr_fp32.engine')
 p.add_argument('--tensorrt-root',type=Path,default=Path(os.environ.get('TENSORRT_ROOT',str(Path.home()/'opt/tensorrt/TensorRT-10.14.1.48'))))
 p.add_argument('--max-chamfer-mm',type=float,default=.15)
 p.add_argument('--config',type=Path,default=(ROOT.parent/'training')/'cfgs/MuJoCo_models/train.yaml')
 p.add_argument('--checkpoint',type=Path,default=(ROOT.parent/'training')/'experiments/train/MuJoCo_models/mujoco_v8_far_finetune/ckpt-best.pth')
 a=p.parse_args()
 a.source=a.source.resolve();a.engine=a.engine.resolve();a.tensorrt_root=a.tensorrt_root.resolve();a.config=a.config.resolve();a.checkpoint=a.checkpoint.resolve()
 # Original training YAML resolves dataset paths relative to its project directory.
 os.chdir(a.source)
 sys.path.insert(0,str(a.source))
 from tools import builder
 from utils.config import cfg_from_yaml_file
 cfg=cfg_from_yaml_file(str(a.config))
 model=builder.model_builder(cfg.model).cuda().eval()
 builder.load_model(model,str(a.checkpoint))
 data=Path(cfg.dataset.test._base_.DATA_PATH)
 rows=json.loads((data/'test.json').read_text());out=ROOT/'tests/results';out.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(a.tensorrt_root/'lib')+':'+env.get('LD_LIBRARY_PATH','')
 # Choose up to nine samples spread across recorded distances.
 rows=sorted(rows,key=lambda r:(r['pose']['distance_m'],r['pose']['azimuth_deg'],r['pose']['elevation_deg']))
 selected=[rows[i] for i in np.unique(np.linspace(0,len(rows)-1,min(9,len(rows))).round().astype(int))]
 report=[]
 with torch.inference_mode():
  for row in selected:
   partial=np.load(data/row['partial']).astype(np.float32)
   reference=model(torch.from_numpy(partial).unsqueeze(0).cuda())[-1][0].cpu().numpy()
   target=out/f'{row["id"]}.npy'
   command=[str(ROOT/'build/bin/pointr_batch'),'--engine',str(a.engine),'--input',str(data/row['partial']),
            '--output',str(target),'--normalized','--warmup','3','--repeat','10']
   run=subprocess.run(command,env=env,check=True,capture_output=True,text=True)
   result=np.load(target);assert result.shape==(8192,3) and np.isfinite(result).all()
   stats=json.loads(run.stdout.strip().splitlines()[-1])
   radius=row['normalization']['radius_m']
   d1=cKDTree(reference).query(result)[0];d2=cKDTree(result).query(reference)[0]
   chamfer=float((d1.mean()+d2.mean())/2)*radius*1000
   max_coordinate_error=float(np.max(abs(result-reference)))*radius*1000
   assert chamfer<a.max_chamfer_mm,(row['pose'],chamfer)
   # Output tail must preserve all observed input points exactly.
   assert np.array_equal(result[-2048:],partial)
   item=dict(id=row['id'],pose=row['pose'],cross_model_chamfer_mm=chamfer,
             max_indexwise_coordinate_difference_mm=max_coordinate_error,stats=stats)
   report.append(item);print(json.dumps(item),flush=True)
  # Physical-meter input path: verify deterministic normalization/inverse and saved output.
  row=selected[-1];partial=np.load(data/row['partial']);n=row['normalization']
  physical=(partial*n['radius_m']+np.array(n['centroid'])).astype(np.float32)
  path=out/'physical_input.npy';np.save(path,physical)
  target=out/'physical_output.npy'
  run=subprocess.run([str(ROOT/'build/bin/pointr_batch'),'--engine',str(a.engine),'--input',str(path),
       '--output',str(target),'--warmup','0','--repeat','1'],env=env,check=True,capture_output=True,text=True)
  stats=json.loads(run.stdout.strip().splitlines()[-1])
  xyz=((physical-np.array(stats['centroid']))/stats['radius_m']).astype(np.float32)
  ref=model(torch.from_numpy(xyz).unsqueeze(0).cuda())[-1][0].cpu().numpy()*stats['radius_m']+np.array(stats['centroid'])
  result=np.load(target);err=float((cKDTree(ref).query(result)[0].mean()+cKDTree(result).query(ref)[0].mean())/2)*1000
  assert err<a.max_chamfer_mm,err
  assert np.max(abs(result[-2048:]-physical))<1e-6
 report=dict(engine=str(a.engine),gpu=torch.cuda.get_device_name(0),cases=report,physical_input_chamfer_mm=err,
     passed=True,tolerance_mm=a.max_chamfer_mm,
     notes=['Cross-model differences compare native TensorRT with original PyTorch weights, not completion accuracy.',
            'Timing is on this host GPU and includes H2D/inference/D2H. Not an Orin measurement.',
            'Neighbor selection may reorder tied points; Chamfer measures point-set agreement.'])
 (out/'validation.json').write_text(json.dumps(report,indent=2))
 print('Validation passed:',out/'validation.json',flush=True)
if __name__=='__main__':main()
