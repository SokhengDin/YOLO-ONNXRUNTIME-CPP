#include "inference.h"
#include <regex>
#include <iostream>

YOLO_V8::YOLO_V8() : cudaEnable(false), rectConfidenceThreshold(0.6f), iouThreshold(0.5f), resizeScales(1.0f) {}

YOLO_V8::~YOLO_V8() = default;

const char* YOLO_V8::CreateSession(const DLInitParam& iParams) {
    try {
        std::regex pattern("[\u4e00-\u9fa5]");
        if (std::regex_search(iParams.modelPath, pattern)) {
            return "[YOLO_V8]: Your model path is error. Change your model path without Chinese characters.";
        }

        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;
        cudaEnable = iParams.cudaEnable;

        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO_V8");
        Ort::SessionOptions sessionOption;

        if (cudaEnable) {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }

        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

        session = std::make_unique<Ort::Session>(env, iParams.modelPath.c_str(), sessionOption);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        size_t outputNodesNum = session->GetOutputCount();

        std::cout << "Number of input nodes: " << inputNodesNum << std::endl;
        std::cout << "Number of output nodes: " << outputNodesNum << std::endl;

        if (inputNodesNum > 0) {
            auto input_name = session->GetInputNameAllocated(0, allocator);
            std::cout << "Input 0 name: " << input_name.get() << std::endl;
            inputNodeNames.push_back(input_name.get());
        }

        if (outputNodesNum > 0) {
            auto output_name = session->GetOutputNameAllocated(0, allocator);
            std::cout << "Output 0 name: " << output_name.get() << std::endl;
            outputNodeNames.push_back(output_name.get());
        }

        return RET_OK;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return "[YOLO_V8]: Error creating session";
    }
}

const char* YOLO_V8::RunSession(const cv::Mat& iImg, std::vector<DLResult>& oResult) {
    cv::Mat processedImg;
    const char* preprocessResult = PreProcess(iImg, imgSize, processedImg);
    if (preprocessResult != RET_OK) {
        return preprocessResult;
    }

    std::vector<float> inputTensor(processedImg.total() * 3);
    cv::Mat floatImg;
    processedImg.convertTo(floatImg, CV_32F, 1.0 / 255.0);
    cv::split(floatImg, std::vector<cv::Mat>{
        cv::Mat(processedImg.rows, processedImg.cols, CV_32F, inputTensor.data()),
        cv::Mat(processedImg.rows, processedImg.cols, CV_32F, inputTensor.data() + processedImg.total()),
        cv::Mat(processedImg.rows, processedImg.cols, CV_32F, inputTensor.data() + 2 * processedImg.total())
    });

    std::vector<int64_t> inputShape = {1, 3, static_cast<int64_t>(processedImg.rows), static_cast<int64_t>(processedImg.cols)};
    
    return TensorProcess(inputTensor.data(), inputShape, oResult);
}

const char* YOLO_V8::PreProcess(const cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg) {
    cv::cvtColor(iImg, oImg, cv::COLOR_BGR2RGB);
    
    float scaleX = static_cast<float>(iImgSize[1]) / iImg.cols;
    float scaleY = static_cast<float>(iImgSize[0]) / iImg.rows;
    resizeScales = std::min(scaleX, scaleY);

    int newUnpadWidth = static_cast<int>(iImg.cols * resizeScales);
    int newUnpadHeight = static_cast<int>(iImg.rows * resizeScales);

    cv::resize(oImg, oImg, cv::Size(newUnpadWidth, newUnpadHeight));
    
    int padRight = iImgSize[1] - newUnpadWidth;
    int padBottom = iImgSize[0] - newUnpadHeight;

    cv::copyMakeBorder(oImg, oImg, 0, padBottom, 0, padRight, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return RET_OK;
}

template<typename T>
const char* YOLO_V8::TensorProcess(const T* inputData, const std::vector<int64_t>& inputShape, std::vector<DLResult>& oResult) {
    try {
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value inputTensor = Ort::Value::CreateTensor<T>(memoryInfo, const_cast<T*>(inputData),
            inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3], inputShape.data(), inputShape.size());

        const char* input_name = inputNodeNames[0];
        const char* output_name = outputNodeNames[0];
        auto outputTensors = session->Run(Ort::RunOptions{nullptr}, &input_name, &inputTensor, 1, &output_name, 1);

        return PostProcess(outputTensors[0], oResult);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error in TensorProcess: " << e.what() << std::endl;
        return "[YOLO_V8]: Error during tensor processing";
    }
}


const char* YOLO_V8::PostProcess(const Ort::Value& outputTensor, std::vector<DLResult>& oResult) {
    Ort::TypeInfo typeInfo = outputTensor.GetTypeInfo();
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputShape = tensorInfo.GetShape();

    const float* output = outputTensor.GetTensorData<float>();
    size_t dimensionCount = outputShape.size();
    if (dimensionCount != 3) {
        return "[YOLO_V8]: Unexpected output tensor shape";
    }

    int rows = outputShape[1];
    int dimensions = outputShape[2];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < rows; ++i) {
        const float* row = output + i * dimensions;
        float conf = row[4];
        if (conf < rectConfidenceThreshold) continue;

        float x = row[0], y = row[1], w = row[2], h = row[3];
        int left = static_cast<int>((x - w / 2) / resizeScales);
        int top = static_cast<int>((y - h / 2) / resizeScales);
        int width = static_cast<int>(w / resizeScales);
        int height = static_cast<int>(h / resizeScales);

        int class_id = std::distance(row + 5, std::max_element(row + 5, row + dimensions));
        float score = row[class_id + 5] * conf;

        boxes.emplace_back(left, top, width, height);
        scores.push_back(score);
        class_ids.push_back(class_id);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, rectConfidenceThreshold, iouThreshold, indices);

    for (int idx : indices) {
        DLResult result;
        result.classId = class_ids[idx];
        result.confidence = scores[idx];
        result.box = boxes[idx];
        oResult.push_back(result);
    }

    return RET_OK;
}

const char* YOLO_V8::WarmUpSession() {
    cv::Mat dummyInput(imgSize[0], imgSize[1], CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<DLResult> dummyResult;
    return RunSession(dummyInput, dummyResult);
}

// Explicit template
template const char* YOLO_V8::TensorProcess<float>(const float*, const std::vector<int64_t>&, std::vector<DLResult>&);
#ifdef USE_CUDA
template const char* YOLO_V8::TensorProcess<half>(const half*, const std::vector<int64_t>&, std::vector<DLResult>&);
#endif