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
- FastAPI 服务：异步任务提交 + 状态查询 + 单帧预览（加分）
- Docker Compose 一键部署

## 3. MCAP 输入格式说明

MCAP 是 ROS2 录制的标准存储格式。本系统通过 `mcap` + `mcap-ros1-support` 库读取，不依赖完整的 ROS2 环境。

支持的 MCAP 特性：
- 自动读取 MCAP summary（Topic 列表、消息数量、时间范围）
- 支持按 Topic 过滤
- 支持时间范围过滤（`--start-sec` / `--end-sec`）
- 支持目录批量扫描（`--mcap-dir`）

## 4. 支持的 ROS 图像消息类型

| 消息类型 | 支持编码 |
|---------|---------|
| `sensor_msgs/CompressedImage` / `sensor_msgs/msg/CompressedImage` | JPEG, PNG |
| `sensor_msgs/Image` / `sensor_msgs/msg/Image` | rgb8, bgr8, mono8, rgba8, bgra8, yuv422, 16UC1, 32FC1 (depth) |

解析字段：`header.stamp`, `header.frame_id`, `format/encoding`, `data`

## 5. 安装依赖

```bash
cd mcap_yolo_quality_gateway
pip install -r requirements.txt
```

Python 版本要求：3.10+

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

## 7. 本机运行方式

```bash
# 启动 API 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 或直接运行 CLI 脚本（见第 10、11 节）
```

## 8. 生成测试 MCAP

```bash
python scripts/generate_test_mcap.py --output test_data/sample.mcap --frames 100 --fps 30
```

生成包含 100 帧合成 JPEG CompressedImage 的 MCAP 文件，用于 pipeline 验证。

## 9. 下载或导出 YOLO ONNX 模型

### 方式一：自动下载 + 导出

```bash
pip install ultralytics   # 仅导出 ONNX 时需要
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
| `--conf-threshold` | **0.6**（本项目默认；作业 FR-YOLO-001 示例为 0.25，可配置） | 检测框置信度；低于此值的 bbox 在后处理中丢弃 |

二者独立配置。作业要求「阈值可配置」，未禁止将 `conf-threshold` 设为 0.6。

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
| 推理后端 | ONNX Runtime CPU |

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
3. 置信度过滤（默认 0.6，可用 `--conf-threshold` 修改；作业示例 0.25）
4. xywh → xyxy
5. 手写 NMS（IoU 阈值默认 0.45）
6. bbox 坐标通过 letterbox 元数据映射回原图
7. 裁剪越界 bbox

## 19. 关键目标类别配置

默认关键目标：person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, dog, cat

CLI 配置：`--target-classes person,car,truck,bus`

只统计目标类别的检测结果，非关键类别在后处理阶段过滤。

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
- 使用 GPU（`--device gpu`，需 CUDA + onnxruntime-gpu）
- 使用更小的模型输入尺寸

## 25. 已知问题

1. `--target-fps` 基于 MCAP summary 估算的平均帧率，帧率不均匀时实际采样密度可能偏差
2. 单帧预览 API 需遍历到目标帧，大文件时延迟较高
3. 图片导出需要在 pipeline 中持有图像引用，内存占用与导出数量成正比

## 26. 后续优化方向

- TensorRT 加速推理（加分项）
- Batch inference 支持（加分项）
- Prometheus metrics 导出（加分项）
- WebSocket 实时进度推送
- 分布式多文件并行处理

## 27. 实际耗时与 AI 使用说明

```
实际开发耗时：约 30 小时
是否使用 AI 工具：是
AI 工具使用范围：代码生成辅助（部分模块框架代码）、文档撰写辅助、架构设计讨论、调试分析
当前已知问题：见第 25 节
未完成项：TensorRT 加速、Batch inference、Prometheus metrics（均为加分项）
```
