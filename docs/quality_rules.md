# Quality Rules

## Single-Frame Quality Metrics

| Metric | Computation | Threshold |
|--------|-------------|-----------|
| Brightness mean | Mean of grayscale image | < 30 = too dark, > 225 = too bright |
| Brightness std | Std of grayscale image | — |
| Blur score | Variance of Laplacian | < 100 = blurry |
| Contrast score | Std of grayscale (same as brightness_std) | < 20 = low contrast |
| Saturation mean | Mean of HSV S channel | — |
| Solid color | Std < 5 across all channels | Binary flag |
| Channel anomaly | One channel mean deviates > 100 from others | Binary flag |
| Aspect ratio | w/h outside [0.2, 5.0] range | Binary flag |

## Scoring Formula

```
quality_score = 1.0
  - blur_penalty        (max 0.35, linear from threshold to 0)
  - exposure_penalty     (max 0.25, linear for dark or bright)
  - contrast_penalty     (max 0.15, linear from threshold to 0)
  - resolution_penalty   (max 0.15, binary if below min dimensions)
  - corruption_penalty   (1.0, binary → forces score to 0)
  - solid_color          (0.10)
  - channel_anomaly      (0.05)
  - aspect_ratio         (0.10)
```

## Quality Gate

Frames with `quality_score < quality_threshold` (default 0.6) are marked as `bad_quality`.

Default behavior: bad quality frames skip YOLO inference.
Override: `--infer-low-quality true` forces inference on all frames.

## Configurable Thresholds

All thresholds are configurable via `app/core/config.py` or environment variables with `MCAP_` prefix:

- `MCAP_BLUR_THRESHOLD=100`
- `MCAP_BRIGHTNESS_LOW=30`
- `MCAP_BRIGHTNESS_HIGH=225`
- `MCAP_CONTRAST_THRESHOLD=20`
- `MCAP_MIN_WIDTH=64`
- `MCAP_MIN_HEIGHT=64`
- `MCAP_QUALITY_THRESHOLD=0.6`
