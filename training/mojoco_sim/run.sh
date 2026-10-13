#!/usr/bin/env bash
# MuJoCo 仿真入口；使用本目录的独立 Python 环境，不依赖 conda pointr 环境。
# 批量采集：bash mojoco_sim/run.sh batch
# 批量参数：修改 mojoco_sim/config.json 的 batch 区域后，重新运行上述命令。
# 方位角 × 仰角 × 距离逐一组合；当前远距离扩增配置 40 × 5 × 5 = 1000 个视角。
# 输出默认写入 mojoco_sim/outputs/batch_编号，每个视角一个子目录。
# 无有效点的视角会跳过；manifest.json 记录成功视角和跳过原因。
# 自定义输出：bash mojoco_sim/run.sh batch --output /你的/采集目录
# 自定义配置：bash mojoco_sim/run.sh batch --config /你的/config.json
# 手动采集：bash mojoco_sim/run.sh gui（按 S 保存）；不传参数时默认打开该界面。
set -euo pipefail
# 从脚本位置定位仿真目录，因此从其他工作目录启动也能找到环境和程序。
SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 检查仿真专用环境是否已经安装。
if [[ ! -x "$SIM_DIR/.venv/bin/python" ]]; then
  echo "请先执行 bash $SIM_DIR/setup.sh" >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  set -- gui
fi
# 将 batch、gui 等模式以及其余参数原样传给 sim.py。
exec "$SIM_DIR/.venv/bin/python" "$SIM_DIR/sim.py" "$@"
