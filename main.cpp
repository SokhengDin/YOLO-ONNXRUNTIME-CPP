#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <opencv2/opencv.hpp>

void ProcessImage(YOLO_V8* detector, const std::string& imagePath, bool saveOutput = true) {
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Error: Unable to read image: " << imagePath << std::endl;
        return;
    }

    std::cout << "Image size: " << img.size() << std::endl;
    std::cout << "Image type: " << img.type() << std::endl;

    std::vector<DLResult> res;
    std::cout << "Running session..." << std::endl;
    const char* result = detector->RunSession(img, res);
    if (result != nullptr) {
        std::cerr << "Error running session: " << result << std::endl;
        return;
    }
    std::cout << "Session completed. Number of results: " << res.size() << std::endl;

    for (const auto& re : res) {
        cv::RNG rng(cv::getTickCount());
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        cv::rectangle(img, re.box, color, 3);

        float confidence = std::floor(100 * re.confidence) / 100;
        std::cout << std::fixed << std::setprecision(2);
        std::string label = detector->GetClasses()[re.classId] + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        cv::rectangle(
            img,
            cv::Point(re.box.x, re.box.y - 25),
            cv::Point(re.box.x + static_cast<int>(label.length() * 15), re.box.y),
            color,
            cv::FILLED
        );

        cv::putText(
            img,
            label,
            cv::Point(re.box.x, re.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 0, 0),
            2
        );
    }

    if (saveOutput) {
        std::filesystem::path inputPath(imagePath);
        std::filesystem::path outputPath = inputPath.parent_path() / (inputPath.stem().string() + "_output" + inputPath.extension().string());
        cv::imwrite(outputPath.string(), img);
        std::cout << "Output saved to: " << outputPath << std::endl;
    }

    cv::imshow("Result of Detection", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ProcessVideo(YOLO_V8* detector, const std::string& videoPath, bool saveOutput = true) {
    // ... (keep the existing ProcessVideo function)
}

int ReadCocoYaml(YOLO_V8*& p) {
    std::ifstream file("coco.yaml");
    if (!file.is_open()) {
        std::cerr << "Failed to open coco.yaml file" << std::endl;
        return 1;
    }

    std::string line;
    std::vector<std::string> names;
    bool in_names_section = false;
    while (std::getline(file, line)) {
        if (line.find("names:") != std::string::npos) {
            in_names_section = true;
            continue;
        }
        if (in_names_section) {
            if (line.find(':') == std::string::npos) {
                break;
            }
            std::istringstream iss(line);
            std::string key, value;
            std::getline(iss, key, ':');
            std::getline(iss, value);
            names.push_back(value);
        }
    }

    p->SetClasses(names);
    return 0;
}

YOLO_V8* InitializeDetector(const std::string& modelPath, const std::vector<int>& imgSize) {
    YOLO_V8* yoloDetector = new YOLO_V8;
    if (ReadCocoYaml(yoloDetector) != 0) {
        std::cerr << "Failed to read coco.yaml" << std::endl;
        delete yoloDetector;
        return nullptr;
    }
    DLInitParam params;
    params.rectConfidenceThreshold = 0.1f;
    params.iouThreshold = 0.5f;
    params.modelPath = modelPath;
    params.imgSize = imgSize;
    params.modelType = ModelType::YOLO_DETECT_V8;
    params.cudaEnable = false;
    const char* result = yoloDetector->CreateSession(params);
    if (result != nullptr) {
        std::cerr << "Failed to create session: " << result << std::endl;
        delete yoloDetector;
        return nullptr;
    }
    return yoloDetector;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <input_path>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string inputPath = argv[2];

    std::cout << "Model path: " << modelPath << std::endl;
    std::cout << "Input path: " << inputPath << std::endl;

    YOLO_V8* detector = InitializeDetector(modelPath, {640, 640});
    if (detector == nullptr) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return 1;
    }

    if (detector != nullptr) {
        detector->PrintInputNodeNames();
    }

    std::filesystem::path path(inputPath);
    std::string extension = path.extension().string();

    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
        ProcessImage(detector, inputPath);
    } else if (extension == ".mp4" || extension == ".avi" || extension == ".mov") {
        ProcessVideo(detector, inputPath);
    } else {
        std::cerr << "Unsupported file format: " << extension << std::endl;
    }

    delete detector;
    return 0;
}