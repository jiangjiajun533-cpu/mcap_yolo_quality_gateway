# YOLO Deployment

## Model

- **Name**: YOLOv8n
- **Format**: ONNX (opset 12, simplified)
- **Source**: Ultralytics (https://github.com/ultralytics/assets/releases)
- **Input**: `float32 [1, 3, 640, 640]` — RGB, normalized to [0, 1]
- **Output**: `float32 [1, 84, 8400]` — [cx, cy, w, h, 80 class scores] × 8400 anchors
- **Backend**: ONNX Runtime (CPU by default, GPU optional)

## Preprocessing (FR-YOLO-003)

1. **BGR → RGB** conversion
2. **Letterbox resize**: scale to fit 640×640 maintaining aspect ratio, pad with grey (114, 114, 114)
3. **Normalize**: divide by 255.0
4. **Transpose**: HWC → CHW
5. **Batch dimension**: expand to (1, 3, 640, 640)
6. **Save metadata**: `LetterboxMeta` stores scale, pad_left, pad_top for coordinate inversion

This matches the preprocessing used during YOLOv8 training (letterbox + normalization).

## Postprocessing (FR-YOLO-004)

1. **Format detection**: auto-detect YOLOv8 (1, 84, N) vs YOLOv5 (1, N, 85)
2. **Transpose**: (1, 84, 8400) → (8400, 84)
3. **Extract**: cx, cy, w, h + 80 class scores per anchor
4. **Confidence filter**: keep only anchors with max class score ≥ threshold
5. **Convert**: cxcywh → xyxy
6. **Batched NMS**: per-class NMS with configurable IoU threshold
7. **Coordinate mapping**: xyxy in model space → original image pixels via LetterboxMeta
8. **Target class filter**: only keep detections in `--target-classes`
9. **Clip**: ensure bbox stays within image bounds

## NMS Implementation

Hand-written in `app/yolo/nms.py`:
- Per-class grouping (batched NMS)
- Standard IoU = intersection / union
- Greedy selection: keep highest score, suppress overlapping boxes

## Configuration

| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Confidence threshold | 0.25 | `--conf-threshold` |
| NMS IoU threshold | 0.45 | `--nms-threshold` |
| Input size | 640 | `--input-size` |
| Device | cpu | `--device` |
| Target classes | person,bicycle,car,... | `--target-classes` |

## GPU Support

Pass `--device gpu` to use CUDA via onnxruntime-gpu. Requires:
- NVIDIA GPU with CUDA support
- `onnxruntime-gpu` package installed
- CUDA toolkit and cuDNN
