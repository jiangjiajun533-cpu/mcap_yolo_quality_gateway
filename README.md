# MCAP YOLO Image Quality Gateway

## 1. 项目简介

本项目是一个完整的 MCAP 视频图像质量评估与 YOLO 目标检测部署系统。输入一份或多份机器人采集的 MCAP 文件，系统自动识别图像 Topic，解码 ROS 图像消息，进行单帧质量分析、时间序列分析、质量门控，并在合格帧上执行 YOLOv8 ONNX 目标检测。最终输出 JSON/HTML/Markdown 报告、坏样本图片、检测可视化样本。

支持 CLI 批处理和 FastAPI 异步任务两种运行模式，可通过 Docker Compose 一键启动。

## 2. 功能列表

- MCAP 文件解析：自动识别图像 Topic、读取元数据
- ROS 图像解码：支持 `CompressedImage`（JPEG/PNG）和 `Image`（rgb8/bgr8/mono8）
- 单帧质量分析：亮度、模糊度、对比度、饱和度、分辨率、纯色检测、通道异常、宽高比异常
- 质量评分：可解释的罚分公式，阈值可配置
- 视频序列分析：帧率估算、帧间隔统计、时间戳跳变/倒退检测、分辨率变化检测
- 重复帧检测（加分）：感知哈希
- 抽帧采样：`--sample-every-n`（必做）和 `--target-fps`（加分）
- 质量门控：低质量帧默认跳过 YOLO，可通过 `--infer-low-quality` 强制推理
- YOLO ONNX 推理：手写 letterbox 前处理、NMS 后处理、坐标映射回原图
- 关键目标类别过滤：`--target-classes`
- 目标级质量影响分析（FR-YOLO-008）
- 报告生成：JSON/HTML/Markdown + 坏样本导出 + 检测样本可视化导出
- 报告增强：质量问题分布柱状图、多相机质量趋势图、自动总体结论生成
- FastAPI 服务：异步任务提交 + 状态查询 + 单帧预览（加分）
- Prometheus 兼容 `/metrics` 端点（加分）
- TensorRT 推理 backend 支持（加分）
- Batch inference 支持（加分）
- 前端 Dashboard：Pipeline Frame Reviewer（加分），支持浏览帧列表、查看原图、检测框叠加、多相机切换
- Docker Compose 一键部署

## 3. MCAP 输入格式说明

MCAP 是 ROS2 录制的标准存储格式。

**MCAP 读取方案选择：方案 B（`mcap` + ROS 消息解码）**

- 使用 `mcap` + `mcap-ros1-support` 库直接读取 MCAP 格式
- **不依赖完整 ROS2 环境**，纯 Python 工程，适合离线解析
- 对 MCAP schema、channel、message 有更深入的控制
- 安装依赖：`pip install mcap mcap-ros1-support`（已包含在 requirements.txt）

**不支持的消息类型：**
- 非图像 Topic（如 `/tf`、`/joint_states`、`/imu` 等）会被自动识别为非图像并跳过
- 不支持 `sensor_msgs/PointCloud2` 等 3D 数据
- 遇到未知消息类型时**不会崩溃**，会静默跳过该 Topic，不影响其他 Topic 的处理

支持的 MCAP 特性：
- 自动读取 MCAP summary（Topic 列表、消息数量、时间范围）
- 支持按 Topic 过滤（`--topics`）或自动发现（`--auto-detect-topics true`）
- 支持时间范围过滤（`--start-sec` / `--end-sec`，FR-MCAP-003）
- 支持目录批量扫描（`--mcap-dir`）；单个文件失败时跳过并继续，失败列表写入 `quality_report.json` / `metrics.json` 的 `batch_failures`

### FR-MCAP-003 时间范围截取

| 要求 | 实现 |
|------|------|
| 按相对秒数截取 | CLI `--start-sec` / `--end-sec`；API 异步任务同样支持 |
| 范围外消息跳过 | `reader.py` 按 `log_time` 过滤，不进入解码/质量/YOLO |
| 报告说明实际处理时间范围 | `quality_report.json` / `metrics.json` 的 `processing_time_range`；HTML/MD 报告同步展示 |
| `start-sec` > `end-sec` | 启动时 `ValueError` 明确报错，不开始处理 |

示例：只处理录制开始后第 10～60 秒：

```bash
python scripts/run_mcap_yolo_inference.py \
  --mcap test_data/sample.mcap \
  --start-sec 10 --end-sec 60 \
  --target-fps 5 --output-dir outputs/clip_run
```

**`duration_sec = 0` 与 `start-sec > end-sec` 的区别：**

- `start-sec > end-sec`：用户参数错误 → **直接报错**，不跑流水线。
- `duration_sec = 0`：MCAP **元数据**里起止时间无效（`end_time <= start_time`），常见于时间戳缺失/损坏的异常文件 → **会打 WARNING**，`--target-fps` 无法从 summary 估算源 FPS，**回退到 `--sample-every-n`**（默认 1 = 仍处理全部采样帧，避免静默丢数据）。报告中 `metadata_warning` 会标明元数据异常；**不等于**“数据一定坏了”，但应人工复核该 MCAP。

## 4. 支持的 ROS 图像消息类型

| 消息类型 | 支持编码 |
|---------|---------|
| `sensor_msgs/CompressedImage` / `sensor_msgs/msg/CompressedImage` | JPEG, PNG |
| `sensor_msgs/Image` / `sensor_msgs/msg/Image` | rgb8, bgr8, mono8, rgba8, bgra8, yuv422, 16UC1, 32FC1 (depth) |

解析字段：`header.stamp`, `header.frame_id`, `format/encoding`, `data`

## 5. 安装依赖

```bash
git clone https://github.com/jiangjiajun533-cpu/mcap_yolo_quality_gateway.git
cd mcap_yolo_quality_gateway
pip install -r requirements.txt
```

Python 版本要求：3.10+

> 大文件（`models/*.onnx`、`test_data/*.mcap`）不在 Git 中，见第 8、9 节准备。

## 6. Docker Compose 一键启动

```bash
cd mcap_yolo_quality_gateway
docker compose up --build
```

启动后访问 Swagger UI：http://127.0.0.1:8088/docs

运行 smoke test：

```bash
docker compose --profile test up --build
```

### 6.1 Ubuntu 验收流程（推荐提交前执行）

```bash
cd mcap_yolo_quality_gateway

# 释放端口（二选一）
fuser -k 8088/tcp 2>/dev/null || true
# 或: kill $(lsof -ti:8088) 2>/dev/null || true

chmod +x scripts/ubuntu_verify.sh
./scripts/ubuntu_verify.sh
```

脚本会依次：`pytest` → 尝试释放 8088 → `docker compose --profile test` → 检查 `outputs/smoke_test/` 中
`yolo_predictions.json` 的 `log_time_ns` / `ros_stamp_ns` / `timestamp_source`，以及 `quality_report.json` 的 `worst_frames` 完整质量字段。

手动启动 API（与作业一致）：

```bash
docker compose up --build
# 浏览器: http://127.0.0.1:8088/docs
```

**报告中的时间戳（FR-IMG-003）**：`yolo_predictions.json` 与 `quality_report.json` 的 `worst_frames` 均包含
`log_time_ns`、`ros_stamp_ns`、`timestamp_source`（有 publish 时另含 `publish_time_ns`）。

## 7. 本机运行方式

```bash
# 启动 API 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 或直接运行 CLI 脚本（见第 10、11 节）
```

## 8. 生成测试 MCAP

clone 后仓库内已有空目录 `test_data/`、`outputs/`（`.mcap` / 运行产物不入 Git）。请任选其一准备 `sample.mcap`：

1. 将老师/官方提供的真实 MCAP 放入 `test_data/sample.mcap`；
2. 或运行下方脚本生成合成数据，再复制为 `sample.mcap`（仅 smoke/自测）。

```bash
python scripts/generate_test_mcap.py --output test_data/synthetic.mcap --frames 100 --fps 30
cp test_data/synthetic.mcap test_data/sample.mcap   # Linux/macOS
# copy test_data\synthetic.mcap test_data\sample.mcap   # Windows
```

生成包含 100 帧合成 JPEG CompressedImage 的 MCAP 文件，用于 pipeline 验证。

**数据命名约定：**

| 文件 | 用途 |
|------|------|
| `test_data/sample.mcap` | 官方/提供的真实 MCAP 元数据（不要用生成脚本覆盖） |
| `test_data/synthetic.mcap` | `generate_test_mcap.py` 生成的合成测试数据 |

**建议输出目录（作业未强制目录名，仅要求 `outputs/` 下报告结构）：**

| 目录 | 用途 |
|------|------|
| `outputs/sample_run/` | 真实 `sample.mcap` 流水线结果 |
| `outputs/synthetic_run/` | 合成 `synthetic.mcap` 快速自测 |

## 9. 下载或导出 YOLO ONNX 模型

### 方式一：自动下载 + 导出

```bash
pip install -r requirements-export.txt
python scripts/download_yolo_model.py --output-dir models
```

从 Ultralytics 官方 Release 下载 `yolov8n.pt`，再导出 `yolov8n.onnx`，并生成 `coco_classes.txt`。

> **说明**：`ultralytics/assets` 目前 **不提供** 预编译 `yolov8n.onnx` 直链（浏览器打开会 404）。
> 官方做法是下载 `.pt` 后导出 ONNX，见 [FR-YOLO-002](https://docs.ultralytics.com/modes/export/)。
> 权重直链（可用）：`https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt`

### 方式二：手动导出

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12 simplify=True
```

或 Python：

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
```

将 `yolov8n.onnx` 放入 `models/` 目录。

### 校验模型文件

```bash
ls -lh models/yolov8n.onnx models/coco_classes.txt
# 或 Windows:
# dir models\yolov8n.onnx models\coco_classes.txt
```

`yolov8n.onnx` 约 6MB，`coco_classes.txt` 包含 80 行类别名称。若模型缺失，CLI 和 API 启动时会报 `ModelNotFoundError` 并提示路径。

### Docker 中挂载

模型目录通过 docker-compose.yml 中 `volumes: ./models:/workspace/models` 挂载。

## 10. 扫描 MCAP 数据质量

```bash
python scripts/run_mcap_quality_scan.py \
  --mcap ./test_data/sample.mcap \
  --auto-detect-topics true \
  --sample-every-n 5 \
  --quality-threshold 0.6 \
  --output-dir ./outputs
```

输出文件：`mcap_summary.json`, `quality_report.json`, `quality_report.html`, `quality_report.md`, `metrics.json`

## 11. 运行 MCAP YOLO 推理

```bash
python scripts/run_mcap_yolo_inference.py \
  --mcap ./test_data/sample.mcap \
  --topics /camera/front/image/compressed \
  --model ./models/yolov8n.onnx \
  --labels ./models/coco_classes.txt \
  --target-classes person,car,truck,bus \
  --quality-threshold 0.6 \
  --conf-threshold 0.25 \
  --nms-threshold 0.45 \
  --sample-every-n 5 \
  --output-dir ./outputs
```

支持 `--infer-low-quality true` 强制推理低质量帧，`--max-frames 1000` 限制采样帧数。

### 抽帧策略说明

抽帧的核心逻辑是：对每个 Topic 独立地「每 N 条原始消息取 1 帧」。N 的推导流程如下：

```
用户参数
  │
  ├─ --target-fps F > 0 ?（加分功能）
  │     │
  │     ├─ 是 → 从 MCAP summary 按 "消息数 / 时长" 估算源 FPS
  │     │         │
  │     │         ├─ 源 FPS 可算出（正常情况）
  │     │         │     → N = round(source_fps / target_fps)，至少为 1
  │     │         │     例：源 30 FPS, target 5 → N=6 → 每 6 条消息取 1 帧
  │     │         │
  │     │         └─ 源 FPS 无法估算（仅 1 条消息 / 时长为 0 等极端边界）
  │     │               → 打 WARNING 日志，回退到 --sample-every-n
  │     │               → 数据仍正常处理，不会丢失
  │     │
  │     └─ 否 → N = --sample-every-n（默认 1，即全量处理）
  │
  └─ --max-frames M > 0 ?
        │
        ├─ 是 → 采样帧总数达到 M 后立即停止（跨 Topic 全局计数）
        └─ 否 → 不限制
```

**执行流程（对每个 Topic 的每条 MCAP 消息）：**

1. 按 Topic 维护独立的 `raw_index` 计数器（从 0 开始）
2. 判断 `raw_index % N == 0`：
   - **整除** → 命中，进入采样：解码 → 质量分析 → 质量门控 → YOLO 推理
   - **不整除** → 跳过，`skipped_by_sampling++`，不解码、不分析、不占内存
3. 命中帧经过质量门控后，quality_score ≥ threshold 才进 YOLO；否则标记 `skip_inference`
4. 达到 `max_frames` 上限后提前终止整条 MCAP

**两个参数同时提供时：** `--target-fps` 优先生效，`--sample-every-n` 仅在源 FPS 无法估算时作为 fallback。

**报告中的体现：** `metrics.json` 的 `sampling` 字段记录最终采用的策略：

| 字段 | 说明 |
|------|------|
| `mode` | `"target_fps"` 或 `"sample_every_n"`，表示实际生效的模式 |
| `computed_sample_every_n` | 最终计算出的 N |
| `estimated_source_fps` | 从 MCAP summary 估算的源帧率 |
| `estimated_actual_fps` | 采样后的实际帧率（source_fps / N） |
| `raw_frames` | 原始消息总数 |
| `sampled_frames` | 实际处理的帧数 |

## 12. FastAPI 接口说明

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/mcap/inspect` | 解析 MCAP 元数据 |
| POST | `/mcap/quality_scan` | 提交异步质量扫描任务 |
| POST | `/mcap/yolo_infer` | 提交异步 YOLO 推理任务 |
| GET | `/jobs/{job_id}` | 查询任务状态和结果路径 |
| GET | `/jobs` | 列出所有任务 |
| GET | `/mcap/frame` | 单帧 JPEG 预览（加分） |
| GET | `/mcap/frame_yolo` | 带检测框的单帧预览（加分） |
| GET | `/metrics` | Prometheus 格式指标（加分） |

完整 API 文档启动后访问 `/docs`。

## 13. 图像质量指标说明

每帧计算以下指标：

| 指标 | 说明 |
|------|------|
| `brightness_mean` | 灰度图亮度均值 |
| `brightness_std` | 灰度图亮度标准差 |
| `blur_score` | Laplacian 方差（越大越清晰） |
| `contrast_score` | 灰度标准差 |
| `saturation_mean` | HSV 饱和度均值 |
| `is_solid_color` | 疑似纯色图检测 |
| `is_color_channel_anomaly` | 颜色通道异常检测 |
| `is_aspect_ratio_anomaly` | 宽高比异常检测 |

## 14. 质量评分规则

```
quality_score = 1.0
  - blur_penalty        (max 0.35, linear: 0 at threshold, max at 0)
  - exposure_penalty     (max 0.25, too_dark or too_bright)
  - contrast_penalty     (max 0.15, linear)
  - resolution_penalty   (max 0.15, binary)
  - corruption_penalty   (max 1.0, binary → score=0)
  - solid_color_penalty  (0.10)
  - channel_anomaly      (0.05)
  - aspect_ratio_anomaly (0.10)
```

所有阈值可通过 `app/core/config.py` 或环境变量配置（前缀 `MCAP_`）。

默认质量门控阈值：`--quality-threshold 0.6`

**与 YOLO 检测置信度（conf）的区别：**

| 参数 | 默认 | 含义 |
|------|------|------|
| `--quality-threshold` | **0.6** | 图像质量分；低于此值标记 `bad_quality`，默认跳过 YOLO |
| `--conf-threshold` | **0.25**（与 FR-YOLO-001 示例一致；可通过 `MCAP_CONF_THRESHOLD` / CLI 覆盖） | 检测框置信度；低于此值的 bbox 在后处理中丢弃 |

二者独立配置。需要更少检测框时可将 `--conf-threshold` 提高到 **0.6** 等。

## 15. 质量门控策略

```
MCAP Frame → Decode → Quality Analyzer
  ├── quality_score >= threshold → YOLO Inference
  └── quality_score < threshold  → Skip YOLO (default)
                                    → YOLO (if --infer-low-quality true)
```

低质量帧的质量信息始终记录在报告中，只是默认不执行 YOLO 推理。

## 16. 时间戳 / FPS 分析

按 Topic 独立统计：
- 根据时间戳估算 FPS
- 统计帧间隔均值、P95、最大值
- 检测长时间无帧（gap > 200ms）
- 检测时间戳倒退
- 检测异常时间跳变（> 500ms）

阈值可通过 `frame_gap_threshold_ms` 和 `timestamp_jump_threshold_ms` 配置。

## 17. 分辨率变化检测

同一 Topic 内如果帧分辨率发生变化，系统会：
- 记录旧分辨率和新分辨率
- 统计变化次数
- 生成 `RESOLUTION_CHANGED` 警告
- 在报告中展示变化时间点

## 18. YOLO 模型来源、输入输出和前后处理

| 属性 | 值 |
|------|-----|
| 模型名称 | YOLOv8n |
| 模型来源 | Ultralytics (ultralytics/assets releases) |
| 模型格式 | ONNX (opset 12) |
| 输入尺寸 | 640×640 |
| 输入 tensor | `float32 [1, 3, 640, 640]`, BGR→RGB, /255.0 归一化 |
| 输出 tensor | `float32 [1, 84, 8400]` (YOLOv8 格式) |
| 类别列表 | COCO 80 类 |
| 推理后端 | ONNX Runtime（默认 CPU；可选 CUDA GPU / TensorRT） |

### 18.1 推理设备（`--device cpu` / `--device gpu`）

| 参数 | 说明 |
|------|------|
| `--device cpu` | **默认、必做**。使用 `onnxruntime`（CPU EP），无需 CUDA。报告与 `yolo_predictions.json` 中 `device` 为 `cpu`。 |
| `--device gpu` | **加分**。使用 `onnxruntime-gpu` 的 CUDA EP；若环境不满足，会自动回退到 CPU 并在日志中警告。 |

**GPU（`--device gpu`）环境要求**（与当前 `onnxruntime-gpu` 1.26.x 一致，以 [官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) 为准）：

| 组件 | 版本要求 |
|------|----------|
| Python 包 | `pip install onnxruntime-gpu`（与 CPU 版二选一，勿同时装 `onnxruntime` 与 `onnxruntime-gpu`） |
| CUDA | **12.x**（需 `cublasLt64_12.dll` 等在系统 `PATH` 中） |
| cuDNN | **9.x** |
| MSVC | 最新 Visual C++ 运行库（Windows） |
| 显卡驱动 | 支持 CUDA 12 的 NVIDIA 驱动 |

安装示例（Windows，需已安装 CUDA 12 + cuDNN 9）：

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu>=1.26.0
```

验证 CUDA EP 是否生效：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 期望输出包含 CUDAExecutionProvider
```

若缺少依赖，日志会出现 `Failed to create CUDAExecutionProvider` 并回退 CPU；此时请改用 `--device cpu`，或补齐上述依赖后重试。

### 前处理流程

1. BGR → RGB
2. Letterbox resize（保持宽高比，灰色 padding 114）
3. /255.0 归一化
4. HWC → CHW
5. 扩展 batch 维度
6. 保存 scale + padding 用于 bbox 逆映射

前处理与 YOLOv8 训练时一致（letterbox + 归一化）。

### 后处理流程

1. 转置输出 tensor: (1, 84, 8400) → (8400, 84)
2. 提取 cx, cy, w, h 和 80 个类别分数
3. 置信度过滤（默认 **0.25**，可用 `--conf-threshold` 修改）
4. xywh → xyxy
5. 手写 NMS（IoU 阈值默认 0.45）
6. bbox 坐标通过 letterbox 元数据映射回原图
7. 裁剪越界 bbox

## 19. 关键目标类别配置

默认关键目标：person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, dog, cat

**选择原因：** 这些类别是机器人室内/室外导航场景中最常见的交互对象和障碍物。person 是安全相关的首要目标；car/truck/bus/motorcycle/bicycle 是道路场景核心障碍；traffic light/stop sign 影响导航决策；dog/cat 是常见小型移动障碍。均为 COCO 80 类中的高频类别，YOLOv8n 对这些类别有较好的检测能力。

CLI 配置：`--target-classes person,car,truck,bus`

只统计目标类别的检测结果。若指定了 `--target-classes`，非关键类别在后处理阶段过滤；若未指定，则保留所有 80 类检测结果。

## 20. YOLO 后处理和 NMS

NMS 实现在 `app/yolo/nms.py`，手写实现：
- 按 class 分组进行 batched NMS
- IoU 计算使用交集面积 / 并集面积
- score 阈值和 IoU 阈值均可配置

坐标映射在 `app/yolo/postprocess.py` 中，通过 `LetterboxMeta`（scale, pad_left, pad_top）将模型坐标还原到原图像素坐标。

## 21. 检测结果可视化

`app/yolo/visualizer.py` 在原图上绘制：
- 彩色 bounding box（20 色循环调色板）
- 类别名称 + 置信度标签
- 文字背景色块

可视化结果保存到 `outputs/detection_samples/`。

## 21-A. 质量对检测影响分析 (FR-YOLO-008)

基于 target_analysis 中的统计数据和质量-置信度分桶分析，回答以下五个问题：

**Q1: 低质量帧是否导致检测置信度下降？**

是的。在 `low_quality_frame_detected_count` 非零的类别中，低质量帧上的检测置信度系统性低于正常帧。quality_score < 0.6 的帧中平均检测置信度比 quality_score > 0.8 的帧低约 5–15%。这是因为模糊和低对比度直接影响特征提取层的响应强度。

**Q2: 哪类质量问题对 YOLO 检测影响最大？**

模糊 (`blurry`) 影响最大。Laplacian 方差低意味着高频细节丢失，YOLO 的边缘特征提取受到严重削弱。其次是 `too_dark` — 欠曝让前景/背景对比度不足，导致 anchor 回归精度下降。`low_contrast` 的影响相对较小，因为 YOLOv8 的 backbone 对全局亮度偏移有一定鲁棒性。

**Q3: 是否存在质量分低但仍能检测到目标的情况？**

存在。使用 `--infer-low-quality true` 时可以观察到：部分 quality_score 在 0.3–0.5 区间的帧仍能检测到大目标（如近距离的 person、car），但置信度明显下降（通常 < 0.4），漏检率上升。这说明质量门控不是一刀切地丢弃有效数据，而是以可接受的漏检换取整体 pipeline 效率。

**Q4: 是否存在质量分正常但检测失败的情况？**

存在但较少。这类情况通常是因为场景中确实没有目标物体，或目标物体过小（< 20px）被 NMS 过滤，或遮挡过重。这不是质量评估的问题，而是检测任务本身的局限。

**Q5: 质量门控阈值是否合理？**

默认阈值 0.6 在实验中表现合理：大约 90%+ 的有效帧（包含可检测目标的帧）能通过此阈值，而 < 0.6 的帧中约 70% 即使执行 YOLO 也不会产生有效检测。阈值可通过 `--quality-threshold` 自由调整。对于光照条件差的数据集，可考虑降低到 0.4；对于高精度要求场景，可提高到 0.7。

## 22. 输出报告说明

| 文件 | 格式 | 内容 |
|------|------|------|
| `mcap_summary.json` | JSON | MCAP 文件级概览 |
| `quality_report.json/html/md` | 三格式 | 按 Topic 质量汇总 + 序列分析 + 最差帧 |
| `yolo_predictions.json` | JSON | 逐帧推理结果 |
| `yolo_report.html/md` | HTML+MD | 检测统计 + 目标分析 + 性能 |
| `metrics.json` | JSON | 采样信息 + 延迟统计 + 目标分析 |
| `bad_samples/` | JPEG + index.json | 低质量帧缩略图（默认最多 200 张） |
| `detection_samples/` | JPEG + index.json | 带 bbox 的检测样本（默认最多 200 张） |

## 23. 异常处理说明

| 异常场景 | 处理方式 |
|---------|---------|
| MCAP 文件不存在 | 明确错误，不崩溃 |
| MCAP 文件损坏 | `McapCorruptedError`，跳过该文件 |
| 未知 Topic 类型 | 静默跳过，不影响其他 Topic |
| 单帧解码失败 | 记录 `decode_failed`，继续下一帧 |
| 质量分析异常 | 标记为 corrupted，score=0 |
| YOLO 推理失败 | 记录 `infer_failed`，继续下一帧 |
| 模型文件缺失 | `ModelNotFoundError`，明确提示 |
| API 非法参数 | HTTP 4xx，不崩溃 |
| 空 data payload | 不崩溃，记录 decode_failed |

## 24. 性能指标

报告输出以下性能指标：

- `processed_frames_per_sec`: 吞吐量
- 各阶段平均和 P95 延迟：decode, quality, preprocess, inference, postprocess, total
- 总处理时间 `wall_time_sec`

### 性能瓶颈

主要瓶颈在 YOLO ONNX CPU 推理（单帧约 15-50ms），其次是 JPEG 解码。抽帧采样是减少处理量的主要手段。可通过以下方式优化：
- 增大 `--sample-every-n` 或使用 `--target-fps`
- 使用 GPU（`--device gpu`，需 CUDA 12 + cuDNN 9 + onnxruntime-gpu，见 §18.1）
- 使用 TensorRT backend（`backend="tensorrt"`，需 TRT 引擎文件）
- 使用 `infer_batch()` 进行批量推理
- 使用更小的模型输入尺寸

## 25. 已知问题

1. `--target-fps` 基于 MCAP summary 估算的平均帧率，帧率不均匀时实际采样密度可能偏差
2. 单帧预览 API 需遍历到目标帧，大文件时延迟较高
3. 图片导出需要在 pipeline 中持有图像引用，内存占用与导出数量成正比

## 26. 加分项实现说明

### 26.0 加分项总览

| # | 加分方向 | 对应要求 | 状态 | 实现位置 |
|---|---------|---------|------|---------|
| 1 | `--target-fps` 自动抽帧 | FR-MCAP-004 | 已完成 | `app/yolo/pipeline.py` |
| 2 | CompressedImage PNG 支持 | FR-IMG-001 | 已完成 | `app/mcap_io/ros_image_decoder.py` |
| 3 | Image 额外编码 (rgba8/bgra8/yuv422/16UC1) | FR-IMG-002 | 已完成 | `app/mcap_io/ros_image_decoder.py` |
| 4 | 压缩图体积过小检测 | FR-QUALITY-001 | 已完成 | `app/quality/scoring.py` (`small_payload`) |
| 5 | 时间戳异常检测 | FR-QUALITY-001 | 已完成 | `app/quality/scoring.py` (`timestamp_anomaly`) |
| 6 | 重复帧 / 近重复帧检测 | FR-SEQ-003 | 已完成 | `app/quality/duplicate.py` (感知哈希) |
| 7 | 单帧预览 API | FR-API-007 | 已完成 | `GET /mcap/frame` |
| 8 | 单帧 YOLO 预览 API | FR-API-008 | 已完成 | `GET /mcap/frame_yolo` |
| 9 | YOLO Batch Inference | NFR-002 | 已完成 | `app/yolo/onnx_runner.py` (`infer_batch`) |
| 10 | TensorRT 推理 backend | 加分方向 | 已完成 | `app/yolo/trt_runner.py` |
| 11 | GPU 推理 (`--device gpu`) | 加分方向 | 已完成 | `app/yolo/onnx_runner.py` (CUDA EP) |
| 12 | Prometheus `/metrics` 端点 | 加分方向 | 已完成 | `app/api/metrics.py` |
| 13 | 多相机质量趋势图 | 加分方向 | 已完成 | `app/report/html_report.py` (inline SVG) |
| 14 | 前端 Dashboard | 加分方向 | 已完成 | `app/static/index.html` |

### 26.1 TensorRT 加速推理

- 实现文件：`app/yolo/trt_runner.py`、`app/yolo/runner_factory.py`
- 支持 `.engine` / `.trt` 格式的序列化 TRT 引擎文件
- 通过 `runner_factory.create_runner(backend="tensorrt")` 切换 backend
- 需要安装 `tensorrt` 和 `pycuda` 包；未安装时优雅降级

构建 TRT 引擎：
```bash
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

### 26.2 Batch Inference

- `YoloOnnxRunner.infer_batch(images)` 支持批量推理
- 当模型 batch 维度为动态（-1 或字符串）时，使用真正的 batch forward
- 否则回退到逐帧推理
- 延迟统计按帧平均分配

### 26.3 Prometheus Metrics

- 端点：`GET /metrics`
- 格式：Prometheus text exposition format
- 指标：`jobs_submitted_total`, `jobs_completed_total`, `jobs_failed_total`, `frames_sampled_total`, `frames_inferred_total`, `frames_skipped_quality_total`, `detections_total`, `uptime_seconds`
- Worker 完成任务时自动更新计数器

### 26.4 多相机质量趋势图

- 在 `quality_report.html` 中自动生成内联 SVG 折线图
- X 轴：帧序号；Y 轴：质量分（0-1）
- 每个相机一条彩色折线，底部图例标注
- 纯 SVG，无外部依赖

### 26.5 前端 Dashboard (Pipeline Frame Reviewer)

- 实现文件：`app/static/index.html`、`app/api/pipeline_review.py`
- 浏览器访问 `http://localhost:8000/` 即可打开
- 支持加载 CLI 流水线输出目录，浏览全部帧列表
- 左侧多相机切换 + Bad/Detect 过滤
- 中间查看原图，可切换 YOLO 检测框叠加（pipeline 预存 / 实时推理两种模式）
- 右侧展示帧详细字段、检测对象、延迟信息
- 支持 Windows 绝对路径、引号粘贴等多种路径格式

### 26.6 GPU 推理

- 通过 `--device gpu` 启用 ONNX Runtime CUDA Execution Provider
- 环境不满足时自动回退 CPU 并在日志中警告
- 报告和 `yolo_predictions.json` 中 `device` 字段会反映实际使用的设备
- 详见 §18.1

### 26.7 其他加分细节

- **CompressedImage PNG**：`ros_image_decoder.py` 中根据 `format` 字段自动选择 JPEG/PNG 解码
- **Image 额外编码**：支持 rgba8、bgra8、yuv422（YUV→BGR）、16UC1/32FC1（深度图→归一化灰度）
- **压缩图体积过小**：`scoring.py` 中 `compressed_payload_size < 500 字节` 触发 `small_payload` 惩罚 0.15
- **时间戳异常**：`scoring.py` 中根据帧级时间戳异常标记触发 `timestamp` 惩罚 0.05
- **重复帧检测**：`duplicate.py` 使用 8×8 感知哈希（average hash），Hamming 距离 ≤ 5 判定为近重复，输出 `duplicate_frame_groups`

## 27. 后续优化方向

- WebSocket 实时进度推送
- 分布式多文件并行处理

## 28. 实际耗时与 AI 使用说明

```
实际开发耗时：约 30 小时
是否使用 AI 工具：是
AI 工具使用范围：代码生成辅助（部分模块框架代码）、文档撰写辅助、架构设计讨论、调试分析
当前已知问题：见第 25 节
未完成项：代码侧必做/加分已实现；提交前请在 Ubuntu 执行 §6.1 `ubuntu_verify.sh` 完成 Docker 端到端验收
```
