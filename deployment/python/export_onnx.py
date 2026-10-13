"""Export pretrained PoinTr; custom FPS nodes are resolved by our TRT plugin."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch,yaml,onnx
from types import SimpleNamespace
from pointr_model import PoinTr
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--checkpoint',type=Path,default=(ROOT.parent/'training')/'experiments/train/MuJoCo_models/mujoco_v8_far_finetune/ckpt-best.pth')
 p.add_argument('--config',type=Path,default=(ROOT.parent/'training')/'cfgs/MuJoCo_models/train.yaml')
 p.add_argument('--output',type=Path,default=ROOT/'models/pointr.onnx')
 a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required for original FPS during export')
 cfg=yaml.safe_load(a.config.read_text())['model']
 model=PoinTr(SimpleNamespace(**cfg)).cuda().eval()
 checkpoint=torch.load(a.checkpoint,map_location='cpu')
 state=checkpoint.get('base_model',checkpoint.get('model'))
 if state is None:raise ValueError('Missing model state_dict')
 model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
 class Dense(torch.nn.Module):
  def __init__(self,net):super().__init__();self.net=net
  def forward(self,xyz):return self.net(xyz)[-1]
 torch.manual_seed(42)
 xyz=torch.randn(1,2048,3,device='cuda')*.3
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  torch.onnx.export(Dense(model),xyz,str(a.output),opset_version=17,
    input_names=['partial'],output_names=['complete'],custom_opsets={'pointr':1},
    do_constant_folding=True)
 exported=onnx.load(str(a.output));onnx.checker.check_model(exported)
 fps=[n for n in exported.graph.node if n.domain=='pointr']
 assert len(fps)==2,[(n.op_type,n.name) for n in fps]
 info=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
   model=cfg,input_name='partial',input_shape=[1,2048,3],output_name='complete',output_shape=[1,8192,3],
   frame='camera_optical',units='normalized',custom_plugins=['pointr::FarthestPointSampling:1'],
   parameters=sum(v.numel() for v in model.parameters()),onnx_nodes=len(exported.graph.node),
   note='Architecture preserved. Export removes training losses and unused coarse output; C++ runtime uses TensorRT only.')
 a.output.with_suffix('.json').write_text(json.dumps(info,indent=2))
 print(json.dumps(info,indent=2))
if __name__=='__main__':main()
