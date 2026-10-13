# 本地训练：仿真采集、数据处理、训练与测试

[返回仓库首页](../README.md) · [C++ TensorRT 实机部署](../deployment/README.md)

**本页所有命令均在 `training/` 目录执行。** 从仓库根目录进入：

```bash
cd training
```

## 1. 环境准备

### MuJoCo 仿真

```bash
bash mojoco_sim/setup.sh
```

创建 `mojoco_sim/.venv`，安装 MuJoCo 3.3.7。仿真与 conda 训练环境相互独立。GUI 需要桌面和 OpenGL；无桌面批量采集可使用 EGL。

### PyTorch 训练

本机验证版本：Python 3.9、PyTorch 2.1.2+cu118、torchvision 0.16.2+cu118、nvcc 11.8。已有这些依赖时：

```bash
conda activate pointr
# nvcc 在 conda 内时；系统安装则改成系统 CUDA 开发目录。
export CUDA_HOME="$CONDA_PREFIX"
bash scripts/setup_pointr_env.sh
python -m pip install -r ../deployment/requirements-export.txt
```

新机器可先建立基础环境：

```bash
conda create -n pointr python=3.9 -y
conda activate pointr
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118
# 安装 CUDA 11.8 开发工具（含 nvcc/头文件）后，设置实际路径：
export CUDA_HOME=/usr/local/cuda-11.8
bash scripts/setup_pointr_env.sh
python -m pip install -r ../deployment/requirements-export.txt
```

安装脚本不替换 PyTorch，负责其他依赖和 CUDA 扩展，并检查模型前向、损失与反向传播。PointNet++ 固定提交为 `b5ceb6d9ca0467ea34beb81023f96ee82228f626`，兼容补丁在 `scripts/pointnet2_compat.patch`。首次安装需要联网；更换 GPU、Python 或 PyTorch 后重编译扩展。

## 2. 仿真采集

修改 `mojoco_sim/config.json`，文件内有中文参数说明：

| 参数 | 内容 |
| --- | --- |
| `mesh` / `mesh_scale` | STL 路径相对 config.json；当前 0.001 将毫米转米 |
| `object_rotation_x_deg` | 当前 90°，使 IObject.STL 长边竖直 |
| `camera` | 640×480、内参、深度范围 |
| `batch` | 方位角 × 仰角 × 距离的采集组合 |
| `density` / `sensor_model` | 随机像素采样、噪声、量化、边缘失效和空洞 |

Gemini2 内参：`fx=518.1068725585938`、`fy=518.32470703125`、`cx=322.09161376953125`、`cy=234.98348999023438`。密度采用 1 米处约 3.2 mm 间距的近似；噪声和缺失未按 Gemini2 实测标定。

```bash
# 批量采集；当前配置为 40×5×5=1000 个视角，以远距离扩增为主。
bash mojoco_sim/run.sh batch --output mojoco_sim/outputs/my_collection

# 无桌面时，用此命令代替上面的批量命令：
MUJOCO_GL=egl bash mojoco_sim/run.sh batch --output mojoco_sim/outputs/my_collection

# 手动采集，按 S 保存：
bash mojoco_sim/run.sh gui
```

首次训练建议在 batch 中同时设置近、中、远距离及多个仰角。距离指相机光心到物体包围盒中心的距离，区别于 optical Z 深度。每个视角保存深度、RGB、掩码、可见点云、完整表面真值和元信息；批次 `manifest.json` 记录成功/跳过情况。

## 3. 数据处理：dataset.yaml

只需编辑 `cfgs/MuJoCo_models/dataset.yaml`：

```yaml
source: mojoco_sim/outputs/my_collection
output: data/MuJoCoObject/dataset_my_collection
sim_config: mojoco_sim/config.json
frame: camera
input_points: 2048
gt_points: 8192
min_raw_points: 64
sector_deg: 15
train_ratio: 0.7
val_ratio: 0.15
seed: 54
all_profiles: false
include_demos: false
```

```bash
conda activate pointr
python tools/prepare_mujoco_dataset.py --config cfgs/MuJoCo_models/dataset.yaml
```

路径相对 `training/`。source 可包含多个批次，脚本递归读取并筛选重复、不符合采集配置或无效样本。按方位角扇区整组划分 train/val/test，样本数比例可能与组数比例略有差异。

输入不足 2048 点时重复实际点补足。完整真值采样 8192 点；输入和真值共用输入中心/最大半径归一化，保存逆变换信息。实机 optical 推理使用 `frame: camera`。

**output 必须是新目录**。生成 `partial/`、`complete/`、`train.json`、`val.json`、`test.json` 和自动加载配置 `loader.yaml`；不要手动编辑 loader.yaml。

## 4. 训练：train.yaml

编辑 `cfgs/MuJoCo_models/train.yaml`：

| 参数 | 用途 |
| --- | --- |
| `dataset_path` | 等于 dataset.yaml 的 output |
| `runtime.exp_name` | 实验名称；也可由命令行覆盖 |
| `total_bs` / `max_epoch` / `optimizer.kwargs.lr` | batch、轮数、学习率；max_epoch=149 对应 150 轮 |
| `runtime.start_ckpts` | 微调起始权重；从头训练设为 null |
| `runtime.resume` | 恢复同名实验断点，沿用保存配置 |
| `loss.repulsion` | 排斥约束，缓解生成点聚集；支持权重和预热设置 |

现有 YAML 保留 v8 远距离微调设置，并引用 v7 权重。**首次克隆从头训练需设置 `start_ckpts: null`、`resume: false`，并自行设置训练学习率**；当前 0.000025 是微调学习率。

```bash
python main.py --config cfgs/MuJoCo_models/train.yaml --exp_name my_object
```

输出 `experiments/train/MuJoCo_models/my_object/`：`ckpt-best.pth` 为验证最优权重，`ckpt-last.pth` 为最近断点，`config.yaml` 为训练配置快照。

微调：设置已有 start_ckpts，resume=false，使用新实验名。恢复中断：resume=true、start_ckpts=null，使用原实验名。

## 5. 测试与四栏 HTML

```bash
python tools/visualize_mujoco_test.py \
  --experiment experiments/train/MuJoCo_models/my_object

VIS_DIR=$(ls -dt experiments/train/MuJoCo_models/my_object/test_vis_* | head -1)
xdg-open "$VIS_DIR/interactive.html"
```

默认加载最佳权重，测试全部 test 样本。`test_vis_时间戳/` 内保存交互 HTML、overview.png、metrics.csv、summary.json，以及每个样本的点云和对比图。

页面包含输入、预测、真值、误差叠加，可旋转/缩放/切换样本，并查看角度、距离、CD-L1 毫米指标和 F@2mm。这是同一物体的留出视角测试，不是跨物体泛化测试。

## 6. 将权重交给部署部分

```bash
python ../deployment/python/export_onnx.py \
  --checkpoint experiments/train/MuJoCo_models/my_object/ckpt-best.pth \
  --config experiments/train/MuJoCo_models/my_object/config.yaml
```

默认生成 `../deployment/models/pointr.onnx`。随后按 [部署 README](../deployment/README.md) 编译和构建 engine。

## 最简日常流程

```bash
conda activate pointr
bash mojoco_sim/run.sh batch --output mojoco_sim/outputs/my_collection
nano cfgs/MuJoCo_models/dataset.yaml
python tools/prepare_mujoco_dataset.py --config cfgs/MuJoCo_models/dataset.yaml
nano cfgs/MuJoCo_models/train.yaml
python main.py --config cfgs/MuJoCo_models/train.yaml --exp_name my_object
python tools/visualize_mujoco_test.py --experiment experiments/train/MuJoCo_models/my_object
python ../deployment/python/export_onnx.py \
  --checkpoint experiments/train/MuJoCo_models/my_object/ckpt-best.pth \
  --config experiments/train/MuJoCo_models/my_object/config.yaml
```
