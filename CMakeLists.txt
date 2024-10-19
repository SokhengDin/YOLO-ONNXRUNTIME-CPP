cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME Yolov8OnnxRuntimeCPPInference)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

option(USE_CUDA "Enable CUDA support" OFF)
if (USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
endif ()

set(ONNXRUNTIME_ROOT /opt/onnxruntime)
include_directories(${ONNXRUNTIME_ROOT}/include)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(PROJECT_SOURCES
    inference.cpp
    main.cpp
)

# Add the executable
add_executable(yolov8_onnx_module ${PROJECT_SOURCES})

# Include directories
target_include_directories(yolov8_onnx_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(yolov8_onnx_module PRIVATE ${OpenCV_LIBS})
target_link_libraries(yolov8_onnx_module PRIVATE ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)

if (USE_CUDA)
    target_link_libraries(yolov8_onnx_module PRIVATE ${CUDA_LIBRARIES})
endif ()

# Copy files
configure_file(${CMAKE_SOURCE_DIR}/coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/yolov8n.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolov8n.onnx COPYONLY)