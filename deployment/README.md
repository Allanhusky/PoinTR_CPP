# 实机部署：C++ TensorRT / Orin NX / ROS 2

[返回仓库首页](../README.md) · [本地训练](../training/README.md)

**本页所有命令均在 `deployment/` 目录执行。** 从仓库根目录进入：

```bash
cd deployment
```

推理使用 C++17、TensorRT 和 CUDA，自定义 FPS 插件保持原 PointNet++ 采样逻辑。Python 只负责离线导出、engine 构建辅助、对照和审阅；C++ 推理不需要 Python/PyTorch。

## 1. 主要文件

| 路径 | 内容 |
| --- | --- |
| `include/pointr/pointr_trt.hpp` | 可复用 C++ 推理 API |
| `src/pointr_trt.cpp` | engine/context、CUDA 缓冲区、推理和坐标还原 |
| `src/fps_plugin.cpp` / `src/fps_cuda.cu` | TensorRT FPS 插件 |
| `src/pointr_batch.cpp` | NPY / ASCII PLY / XYZ 推理入口 |
| `src/pointr_benchmark.cpp` | 持续推理测速 |
| `src/pointr_node.cpp` | 可选 ROS 2 PointCloud2 节点 |
| `python/` | ONNX 导出、engine 构建、测速与可视化 |
| `models/` | ONNX、engine 及生成元信息，本地保留 |
| `examples/` | 小型点云示例 |
| `outputs/` | 预测点云、HTML、速度报告 |

## 2. 准备 ONNX

在有 PyTorch/PointNet++ 的训练主机执行：

```bash
conda activate pointr
python python/export_onnx.py \
  --checkpoint ../training/experiments/train/MuJoCo_models/my_object/ckpt-best.pth \
  --config ../training/experiments/train/MuJoCo_models/my_object/config.yaml
```

输出 `models/pointr.onnx` 和 `models/pointr.json`。固定 batch=1，输入 `[1,2048,3]`、输出 `[1,8192,3]`。部署保留网络结构，移除训练损失及未使用输出分支。

## 3. 编译和生成 engine

需要完整 CUDA/TensorRT 开发头文件、库及 trtexec；仅有 TensorRT Python 包可能不够。本机验证 TensorRT 10.14.1，代码包含 8.5/8.6 分支但需目标验证；TensorRT 11 需将 FPS 迁移为 V3 插件。

本机 RTX 4070 Ti SUPER 示例：

```bash
conda activate pointr
export CUDA_HOME="$CONDA_PREFIX"   # 若 nvcc 在系统 CUDA，改成实际目录
export TENSORRT_ROOT="$HOME/opt/tensorrt/TensorRT-10.14.1.48"
export CUDA_ARCHITECTURES=89
bash scripts/build.sh
python3 python/build_engine.py --tensorrt-root "$TENSORRT_ROOT"
```

Orin NX 示例：将 deployment 源码和 ONNX 复制到设备，在设备重新编译：

```bash
export CUDA_HOME=/usr/local/cuda
export TENSORRT_ROOT=/usr
export CUDA_ARCHITECTURES=87
bash scripts/build.sh
python3 python/build_engine.py --tensorrt-root /usr
```

不要将主机 build、venv、engine 作为 Orin 可执行产物。构建脚本只用 Python 标准库，目标设备不需要训练 conda 环境。默认 FP32；FP16 可通过 `--fp16 --engine models/pointr_fp16.engine` 构建，但需要另做精度对照。

## 4. 推理

```bash
bash run.sh --engine models/pointr_fp32.engine \
  --input examples/partial_camera_far.npy --output outputs/completion.ply
```

输入为**单个目标、相机 optical 坐标、单位米**。程序采样到 2048 点（不足则重复实际点）、归一化、传输、推理，再将 8192 个点还原为米制坐标并写出。已归一化输入加 `--normalized`，必须恰好 2048 点，输出也保持归一化。

实时程序创建一次 `PoinTrTRT`，反复调用 `infer()`，不要每帧加载模型。同一 context/stream 不能并发调用，多线程需分别创建实例。

## 5. 对照、速度和 HTML 审阅

以下辅助工具在训练主机的 pointr 环境执行：

```bash
# 与当前 PyTorch 权重对照
python tests/verify_cpp.py \
  --config ../training/experiments/train/MuJoCo_models/my_object/config.yaml \
  --checkpoint ../training/experiments/train/MuJoCo_models/my_object/ckpt-best.pth

# 使用训练集测速
python python/benchmark_training.py \
  --dataset ../training/data/MuJoCoObject/dataset_my_collection

# 输入/预测 PNG
python python/visualize_cloud.py
```

测速结果在 `outputs/speed_benchmark/summary.json` 和逐帧 CSV，包含均值、P95/P99。模型只加载一次、预热 50 次，数据预读入内存。总计时包含采样处理、归一化、H2D、推理、D2H、还原；训练输入已为 2048 点，不含数万原始点下采样、文件 IO、相机、YOLOE 和 ROS。

本机 RTX 4070 Ti SUPER、v8 的 1710 个训练样本、FP32 米制模式均值约 2.939 ms，约 340 帧/秒；这不是 Orin 性能。

已保留 v8 数据和测试结果的本机可生成四栏离线 HTML：

```bash
python python/visualize_html.py
xdg-open outputs/html_review/interactive.html
```

该示例页面依赖 v8 配对数据和 `tests/results/validation.json`，源码克隆不附带这些数据/历史预测。新训练的完整 test 集 HTML 使用 [训练部分](../training/README.md) 的测试脚本。

## 6. ROS 2 / YOLOE 接入

Orin 上先设置第 3 节 CUDA/TensorRT 环境变量，再执行 ROS 2 Humble 示例：

```bash
source /opt/ros/humble/setup.bash
bash scripts/build.sh -DBUILD_ROS2_NODE=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:$PWD/build/lib:${LD_LIBRARY_PATH:-}"
build/bin/pointr_node --ros-args \
  -p engine_path:="$PWD/models/pointr_fp32.engine" \
  -p input_topic:=/object/points_camera \
  -p expected_frame:=camera_depth_optical_frame \
  -p output_topic:=/pointr/completed_points
```

输入 PointCloud2 为 float32 XYZ、米制、至少 64 个有效点。expected_frame 填相机实际 optical frame，输出保留输入 header。

接入顺序：**Gemini2 深度 + YOLOE 单个目标 mask → optical 目标点云 → PoinTr 补全 → 世界坐标变换**。外部 camera_ws 的点云可能已变换到 map 且合并多个目标，需要在变换前按目标单独发布。本仓库提供补全节点，未修改外部 YOLOE 发布逻辑。

20 Hz 对应每帧 50 ms，整链路速度、共享显存和功耗需要在 Orin 上结合 YOLOE 实测。
