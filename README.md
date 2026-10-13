# PoinTr 点云补全完整工程

本仓库包含从仿真采集、数据处理、模型训练，到 C++ TensorRT 实机推理的完整流程。

```text
PoinTr/
├── training/       # 本地仿真采集、数据制作、PyTorch 训练与测试
│   └── README.md
├── deployment/     # C++ TensorRT、CUDA FPS 插件、ROS 2、导出与测速
│   └── README.md
├── README.md
└── LICENSE
```

| 部分 | 使用说明 | 运行环境 |
| --- | --- | --- |
| 本地训练 | [training/README.md](training/README.md) | MuJoCo 独立 venv；PyTorch conda 环境 `pointr` |
| 实机部署 | [deployment/README.md](deployment/README.md) | C++17、CUDA、TensorRT；ROS 2 可选 |

**推荐顺序：** 在 `training/` 采集并制作数据、训练和测试 → 导出权重为 ONNX → 在目标设备编译 `deployment/` 并生成 TensorRT engine → 接入相机单个目标点云。

当前模型输入 2048 点、输出 8192 点。部署保持原网络结构，推理不依赖 Python/PyTorch。Gemini2 光学坐标系、米制输入需要与训练数据保持一致。Orin NX 上重新编译插件和生成 engine，主机速度不能直接当作 Orin 性能。

已有采集数据、数据集、权重和输出仍保留在本机，并随各自部分移动；Git 默认忽略大文件、虚拟环境和编译结果。GitHub 源码包含 STL、配置和小型点云示例，首次克隆需自行准备数据和权重。

## 源码打包与上传

在仓库根目录执行：

```bash
python training/scripts/package_source.py
# 输出：artifacts/PoinTr_source.tar.gz，不含 .git、数据、权重和 engine

git status --short
git add -A
git diff --cached --stat
# 审阅后自行 commit / push
```

原始 PoinTr 说明保存在 `training/docs/UPSTREAM.txt`，官方数据集说明在 `training/docs/DATASETS.txt`。来源：[yuxumin/PoinTr](https://github.com/yuxumin/PoinTr)。保留根目录 MIT 许可证以及第三方许可证/来源记录。
