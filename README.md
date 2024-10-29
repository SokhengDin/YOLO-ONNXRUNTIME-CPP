# YOLOv8 ONNX C++ Inference

This project implements YOLOv8 object detection using ONNX Runtime in C++. It supports multiple input sources including images, videos, RTSP streams, and Intel RealSense cameras.

## Features

- YOLOv8 object detection with ONNX Runtime
- Support for multiple input types:
  - Single images
  - Video files
  - RTSP streams
  - Directory of images
  - Intel RealSense D435i camera (optional)
- CUDA acceleration support
- Real-time visualization
- Output saving capabilities

## Prerequisites

- Docker
- X11 for display (on Linux)
- CUDA drivers (optional, for GPU support)

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   ├── inference.h
│   ├── input_handler.h
│   └── realsense_handler.h (optional)
├── src/
│   ├── inference.cpp
│   ├── input_handler.cpp
│   └── realsense_handler.cpp (optional)
├── models/
│   └── yolov8n.onnx
├── coco.yaml
└── Dockerfile
```

## Building with Docker

1. Build the Docker image:
```bash
docker build -t onnx-cpp .
```

2. Set up X11 forwarding (Linux):
```bash
export IP=$(ip route get 1.2.3.4 | awk '{print $7}')
xhost +local:docker
```

## Running the Application

### Basic Usage

```bash
docker run -it --rm \
    -e DISPLAY=$IP:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/output:/app/output \
    onnx-cpp /app/build/Yolo8OnnxRuntimeCPPInference <mode> <model> <input> <config>
```

### Command Line Arguments

- `<mode>`: Operation mode (`detect` or `classify`)
- `<model>`: Path to ONNX model file
- `<input>`: Path to input (image/video file, directory, or RTSP URL)
- `<config>`: Path to COCO class names YAML file

### Examples

1. Process a single image:
```bash
docker run -it --rm \
    -e DISPLAY=$IP:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/output:/app/output \
    onnx-cpp /app/build/Yolo8OnnxRuntimeCPPInference detect yolov8n.onnx bus.jpg coco.yaml
```

2. Process a video:
```bash
docker run -it --rm \
    -e DISPLAY=$IP:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/output:/app/output \
    onnx-cpp /app/build/Yolo8OnnxRuntimeCPPInference detect yolov8n.onnx video.mp4 coco.yaml
```

3. Process an RTSP stream:
```bash
docker run -it --rm \
    -e DISPLAY=$IP:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/output:/app/output \
    onnx-cpp /app/build/Yolo8OnnxRuntimeCPPInference detect yolov8n.onnx rtsp://camera_url coco.yaml
```

### Output

- Processed images/videos are saved in the `output` directory
- Detection results are displayed in real-time (when X11 is available)
- Each detection includes:
  - Bounding box
  - Class label
  - Confidence score

## Building without Docker

### Prerequisites

- CMake (>= 3.5)
- OpenCV (>= 4.0)
- ONNX Runtime (1.15.1)
- C++17 compiler
- CUDA Toolkit (optional, for GPU support)

### Build Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create build directory:
```bash
mkdir build && cd build
```

3. Configure and build:
```bash
cmake ..
make -j$(nproc)
```

4. For CUDA support:
```bash
cmake -DUSE_CUDA=ON ..
make -j$(nproc)
```

## Acknowledgments

- YOLOv8 by Ultralytics
- ONNX Runtime by Microsoft
