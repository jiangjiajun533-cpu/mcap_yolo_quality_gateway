# MCAP Input Format

## What is MCAP?

MCAP is a general container format for robotics recordings. ROS 2 `rosbag2` often stores bags as MCAP; ROS 1 data may also appear in MCAP with `message_encoding=ros1`. The file extension does not imply ROS 2 CDR serialization inside.

## Supported Message Types

| Schema name | MCAP `message_encoding` | Supported |
|-------------|-------------------------|-----------|
| `sensor_msgs/CompressedImage` | ros1 | Yes |
| `sensor_msgs/msg/CompressedImage` | ros1 | Yes (schema alias; same decoder path) |
| `sensor_msgs/Image` | ros1 | Yes |
| `sensor_msgs/msg/Image` | ros1 | Yes (schema alias) |
| Any of the above | cdr / ros2 | No (not implemented; needs `mcap-ros2-support` or conversion) |

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

We use `mcap` for the container and `mcap-ros1-support` to deserialize **ros1-encoded** channels, without a full ROS 1/ROS 2 runtime. The reader first loads the summary section (topic list, message counts, time range) without decoding any messages, then iterates decoded messages on demand via `mcap_ros1.decoder.DecoderFactory`.

## Time Range Filtering

Use `--start-sec` and `--end-sec` to process only a portion of the recording. Times are relative to the start of the MCAP file.

## Batch Processing

Use `--mcap-dir` to process all `.mcap` files in a directory recursively.
