#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#define RET_OK nullptr

enum class ModelType {
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
    YOLO_CLS_HALF = 6
};

struct DLInitParam {
    std::string modelPath;
    ModelType modelType = ModelType::YOLO_DETECT_V8;
    std::vector<int> imgSize = {640, 640};
    float rectConfidenceThreshold = 0.6f;
    float iouThreshold = 0.5f;
    int keyPointsNum = 2;
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
};

struct DLResult {
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
};

class YOLO_V8 {
public:
    YOLO_V8();
    ~YOLO_V8();

    const char* CreateSession(const DLInitParam& iParams);
    const char* RunSession(const cv::Mat& iImg, std::vector<DLResult>& oResult);
    const char* WarmUpSession();

    void SetClasses(const std::vector<std::string>& classes) { this->classes = classes; }
    const std::vector<std::string>& GetClasses() const { return classes; }

    void PrintInputNodeNames() const {
        std::cout << "Input node names:" << std::endl;
        for (const auto& name : inputNodeNames) {
            std::cout << " - " << name << std::endl;
        }
    }

private:
    const char* PreProcess(const cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);
    template<typename T>
    const char* TensorProcess(const T* inputData, const std::vector<int64_t>& inputShape, std::vector<DLResult>& oResult);
    const char* PostProcess(const Ort::Value& outputTensor, std::vector<DLResult>& oResult);

    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;
    ModelType modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales;

    std::vector<std::string> classes;
};