# MCAP Input Format

## What is MCAP?

MCAP is the standard recording format for ROS2 bag files. It stores serialized messages from multiple topics in a single file with an index for efficient random access.

## Supported Message Types

| Schema | Encoding | Supported |
|--------|----------|-----------|
| `sensor_msgs/CompressedImage` | ros1 | Yes |
| `sensor_msgs/msg/CompressedImage` | ros2/cdr | Yes |
| `sensor_msgs/Image` | ros1 | Yes |
| `sensor_msgs/msg/Image` | ros2/cdr | Yes |

## CompressedImage Fields

- `header.stamp` → ROS timestamp (sec + nanosec)
- `header.frame_id` → camera frame identifier
- `format` → "jpeg" or "png"
- `data` → compressed image bytes

## Image Fields

- `header.stamp`, `header.frame_id` → same as above
- `encoding` → "rgb8", "bgr8", "mono8", "16UC1"
- `width`, `height` → pixel dimensions
- `data` → raw pixel bytes

## Reading Strategy

We use `mcap` + `mcap-ros1-support` Python packages to read MCAP files without requiring a full ROS2 installation. The reader first loads the summary section (topic list, message counts, time range) without decoding any messages, then iterates decoded messages on demand.

## Time Range Filtering

Use `--start-sec` and `--end-sec` to process only a portion of the recording. Times are relative to the start of the MCAP file.

## Batch Processing

Use `--mcap-dir` to process all `.mcap` files in a directory recursively.
