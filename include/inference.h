#pragma once

#define RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <driver.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

enum MODEL_TYPE 
{
    // Float32 model
    YOLO_DETECT_V8  = 1,
    YOLO_POSE       = 2,
    YOLO_CLS        = 3,

    // Float16 model
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF   = 5,
    YOLO_CLS_HALF       = 6
};

typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType    = YOLO_DETECT_V8;
    std::vector<int> imgSize= { 640, 640 };
    float rectConfidenceThreshold   = 0.6;
    float iouThreshold      = 0.5;
    int keyPointsNum        = 2;
    bool cudaEnable         = false;
    int logSeverityLevel    = 3;
    int intraOpNumThreads   = 1;
} DL_INIT_PARAM;


typedef struct _DL_RESULT 
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;


class YOLO8Onnx
{
    public:
        YOLO8Onnx();

        ~YOLO8Onnx();

    public:
        char* YOLO8Onnx::CreateSession(DL_INIT_PARAM& iParams);

        char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);

        char* WarmUpSession();

        template<typename N>
        char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims, std::vector<DL_RESULT>& oResult);

        char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);

        char* ProcessInput(const std::string& input, std::vector<DL_RESULT>& results);
        

        std::vector<std::string> classes();

    private:
        Ort::Env env;
        Ort::Session* session;

        bool cudaEnable;

        Ort::RunOptions options;

        std::vector<const char*> inputNodeNames;
        std::vector<const char*> outputNodeNames;

        MODEL_TYPE modelType;

        std::vector<int> imgSize;

        float rectConfidenceThreshold;
        float iouThreshold;
        float resizeScales;
};
