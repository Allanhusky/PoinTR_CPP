"""Build a native TensorRT engine using the existing SDK's trtexec."""
import argparse,json,os,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--onnx',type=Path,default=ROOT/'models/pointr.onnx')
 p.add_argument('--engine',type=Path,default=ROOT/'models/pointr_fp32.engine')
 p.add_argument('--plugin',type=Path,default=ROOT/'build/lib/libpointr_plugins.so')
 p.add_argument('--tensorrt-root',type=Path,default=Path(os.environ.get('TENSORRT_ROOT','/usr')))
 p.add_argument('--fp16',action='store_true')
 p.add_argument('--workspace-gb',type=float,default=2)
 a=p.parse_args()
 if a.workspace_gb<=0:raise ValueError('workspace-gb must be positive')
 exe=a.tensorrt_root/'bin/trtexec'
 if not exe.is_file():
  found=shutil.which('trtexec')
  exe=Path(found) if found else Path('/usr/src/tensorrt/bin/trtexec')
  if not exe.is_file():raise FileNotFoundError('Missing trtexec; install the target TensorRT SDK or add trtexec to PATH')
 a.engine.parent.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(a.tensorrt_root/'lib')+':'+env.get('LD_LIBRARY_PATH','')
 help_text=subprocess.run([str(exe),'--help'],env=env,check=True,capture_output=True,text=True).stdout
 plugin_flag='--dynamicPlugins' if '--dynamicPlugins' in help_text else '--plugins'
 cmd=[str(exe),f'--onnx={a.onnx.resolve()}',f'--saveEngine={a.engine.resolve()}',
      f'{plugin_flag}={a.plugin.resolve()}',f'--memPoolSize=workspace:{int(a.workspace_gb*1024)}',
      '--noTF32','--skipInference']
 if a.fp16:cmd.append('--fp16')
 start=time.monotonic();subprocess.run(cmd,env=env,check=True)
 a.engine.with_suffix('.json').write_text(json.dumps(dict(precision='fp16' if a.fp16 else 'fp32',
  sdk_root=str(a.tensorrt_root),engine=str(a.engine),onnx=str(a.onnx),command=cmd,
  build_seconds=time.monotonic()-start,note='Rebuild engine on target GPU; validate FP16 accuracy separately.'),indent=2))
 print('Saved:',a.engine,flush=True)
if __name__=='__main__':main()
