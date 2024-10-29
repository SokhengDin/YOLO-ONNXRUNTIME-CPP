cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME Yolo8OnnxRuntimeCPPInference)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Compile CUDA for FP16 inference if needed
option(USE_CUDA "Enable CUDA support" OFF)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wno-write-strings)
endif()

# ONNXRUNTIME
set(ONNXRUNTIME_VERSION 1.15.1)
if (WIN32)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (UNIX AND NOT APPLE)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (APPLE)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")
else ()
    message(FATAL_ERROR "Unsupported platform. Please set ONNXRUNTIME_ROOT manually.")
endif ()

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add source files
set(PROJECT_SOURCES
    main.cpp
    inference.cpp
    input_handler.cpp 
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (APPLE)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
endif ()

if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif ()

# Copy configuration files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/yolov8n.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolov8n.onnx COPYONLY)

# Create required directories
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/output  # Added output directory
)