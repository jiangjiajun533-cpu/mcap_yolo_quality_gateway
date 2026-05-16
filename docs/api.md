# API Documentation

## Base URL

```
http://127.0.0.1:8000    (local)
http://127.0.0.1:8088    (Docker)
```

Swagger UI: `GET /docs`

---

## GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "mcap_reader_available": true,
  "yolo_backend": "onnxruntime"
}
```

---

## POST /mcap/inspect

Parse MCAP file metadata without processing frames.

**Request:**
```json
{
  "mcap_path": "/workspace/test_data/sample.mcap"
}
```

**Response:**
```json
{
  "mcap_file": "sample.mcap",
  "duration_sec": 60.0,
  "topics": [
    {
      "topic": "/camera/front/image/compressed",
      "message_type": "sensor_msgs/msg/CompressedImage",
      "message_count": 600,
      "is_image_topic": true
    }
  ]
}
```

---

## POST /mcap/quality_scan

Submit an async quality scan job.

**Request:**
```json
{
  "mcap_path": "/workspace/test_data/sample.mcap",
  "topics": ["/camera/front/image/compressed"],
  "sample_every_n": 5,
  "quality_threshold": 0.6
}
```

**Response:**
```json
{
  "job_id": "job-a1b2c3d4",
  "status": "running"
}
```

---

## POST /mcap/yolo_infer

Submit an async YOLO inference job.

**Request:**
```json
{
  "mcap_path": "/workspace/test_data/sample.mcap",
  "topics": ["/camera/front/image/compressed"],
  "model_path": "/workspace/models/yolov8n.onnx",
  "labels_path": "/workspace/models/coco_classes.txt",
  "target_classes": ["person", "car", "truck", "bus"],
  "sample_every_n": 5,
  "quality_threshold": 0.6,
  "conf_threshold": 0.25,
  "nms_threshold": 0.45,
  "infer_low_quality": false
}
```

**Response:**
```json
{
  "job_id": "job-e5f6g7h8",
  "status": "running"
}
```

---

## GET /jobs/{job_id}

Query job status and results.

**Response (running):**
```json
{
  "job_id": "job-e5f6g7h8",
  "status": "running",
  "progress": 0.45
}
```

**Response (finished):**
```json
{
  "job_id": "job-e5f6g7h8",
  "status": "finished",
  "progress": 1.0,
  "result_path": "/workspace/outputs/job-e5f6g7h8/yolo_predictions.json",
  "report_path": "/workspace/outputs/job-e5f6g7h8/yolo_report.html",
  "elapsed_sec": 12.345
}
```

---

## GET /jobs

List all jobs.

---

## GET /mcap/frame (bonus)

Preview a single decoded frame as JPEG.

**Query params:** `mcap_path`, `topic`, `frame_seq`

**Response:** `image/jpeg`

---

## GET /mcap/frame_yolo (bonus)

Preview a single frame with YOLO detections drawn.

**Query params:** `mcap_path`, `topic`, `frame_seq`, `model_path`

**Response:** `image/jpeg`
