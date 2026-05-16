# Architecture

## System Overview

```
MCAP File(s)
    │
    ▼
┌─────────────┐
│  mcap_io/   │  reader, topic_scanner, ros_image_decoder
│  (Layer 1)  │  → FrameRecord per decoded image
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  quality/   │  metrics, scoring, analyzer, sequence_analyzer
│  (Layer 2)  │  → QualityResult per frame
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  yolo/      │  preprocess, onnx_runner, postprocess, nms
│  (Layer 3)  │  → Detection objects per frame
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  pipeline   │  Unified orchestrator: sampling → decode → quality → YOLO
│  (Layer 4)  │  → InferenceRecord stream + PipelineStats
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  report/    │  json_report, html_report, markdown_report, sample_exporter
│  (Layer 5)  │  → JSON / HTML / MD files + image exports
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  scripts/   │     │  api/       │  FastAPI endpoints + async jobs
│  CLI entry  │     │  (Layer 6)  │
└─────────────┘     └─────────────┘
```

## Key Design Decisions

1. **Streaming architecture**: frames are processed one-by-one via generators, never loaded all at once
2. **Per-topic independence**: each topic has separate counters, sequence trackers, and quality aggregators
3. **Unified sampling gate**: both `--sample-every-n` and `--target-fps` resolve to a single n-based sampler
4. **Stats externalization**: `PipelineStats` passed out via `stats_out` parameter for report generation
5. **Thread-safe job management**: `JobManager` uses a lock-protected dict for API async tasks
