#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <random>
#include "inference.h"

// read yaml
std::vector<std::string> ReadClassNames(const std::string& yamlPath) {
    std::ifstream file(yamlPath);

    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << yamlPath << std::endl;
        return {};
    }

    std::vector<std::string> names;
    std::string line;
    bool nameSection = false;

    while (std::getline(file, line)) {
        if (line.find("names:") != std::string::npos) {
            nameSection = true;
            continue;
        }

        if (nameSection && line.find(':') != std::string::npos) {
            std::string name = line.substr(line.find(':') + 1);
            name.erase(0, name.find_first_not_of(" \t"));
            names.push_back(name);
        }
    }

    return names;
}

void VisualizeAndSaveDetection(cv::Mat& img, const std::vector<DL_RESULT>& results, 
                              const std::vector<std::string>& classes, 
                              const std::string& outputPath) {
    // Debug information
    std::cout << "Number of detections: " << results.size() << std::endl;
    
    for (const auto& result: results) {
        // Debug print each detection
        std::cout << "Class: " << classes[result.classId] 
                  << ", Confidence: " << result.confidence 
                  << ", Box: [" << result.box.x << ", " << result.box.y 
                  << ", " << result.box.width << ", " << result.box.height << "]" << std::endl;

        cv::RNG rng(cv::getTickCount());
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        // Draw the rectangle
        cv::rectangle(img, result.box, color, 2);

        // Prepare label
        std::string label = classes[result.classId] + " " + 
                           std::to_string(static_cast<int>(result.confidence * 100)) + "%";

        // Get label size for background rectangle
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Draw label background
        cv::rectangle(img, 
                     cv::Point(result.box.x, result.box.y - labelSize.height - 5),
                     cv::Point(result.box.x + labelSize.width, result.box.y),
                     color, cv::FILLED);

        // Draw label text
        cv::putText(img, label, 
                    cv::Point(result.box.x, result.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, cv::Scalar(0, 0, 0), 1);

        // Debug - verify drawing
        std::cout << "Drew bounding box and label for " << label << std::endl;
    }

    // Create output directory if it doesn't exist
    std::filesystem::path outputDir = std::filesystem::path(outputPath).parent_path();
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directories(outputDir);
        std::cout << "Created output directory: " << outputDir << std::endl;
    }

    // Save the image with debug information
    bool saveResult = cv::imwrite(outputPath, img);
    if (saveResult) {
        std::cout << "Successfully saved result to: " << outputPath << std::endl;
    } else {
        std::cerr << "Failed to save result to: " << outputPath << std::endl;
    }

    // Display the image
    cv::imshow("YOLO8 Result", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void VisualizeAndSaveClassification(cv::Mat& img, const std::vector<DL_RESULT>& results, 
                                  const std::vector<std::string>& classes,
                                  const std::string& outputPath) {
    int positionY = 30;
    for (size_t i = 0; i < std::min(results.size(), size_t(5)); i++) {
        cv::RNG rng(cv::getTickCount() + i);
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        
        std::string label = classes[results[i].classId] + ": " + 
                           std::to_string(static_cast<int>(results[i].confidence * 100)) + "%";
        
        cv::putText(img, label, cv::Point(10, positionY), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        positionY += 30;
    }

    std::filesystem::path outputDir = std::filesystem::path(outputPath).parent_path();
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directories(outputDir);
    }

    cv::imwrite(outputPath, img);
    std::cout << "Saved result to: " << outputPath << std::endl;

    cv::imshow("YOLO8 Result", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <detect/classify> <model_path> <input_path> [yaml_path]" << std::endl;
        return 1;
    }

    std::string task = argv[1];
    std::string modelPath = argv[2];
    std::string inputPath = argv[3];
    std::string yamlPath = (argc > 4) ? argv[4] : "coco.yaml";

    std::filesystem::path inputFilePath(inputPath);
    std::filesystem::path outputDir("/app/output"); 
    std::string outputPath = (outputDir / inputFilePath.filename()).string();
    
    std::string extension = inputFilePath.extension().string();
    outputPath = outputPath.substr(0, outputPath.length() - extension.length()) + 
                "_result" + extension;

    YOLO8Onnx yolo;
    DL_INIT_PARAM params;
    params.modelPath = modelPath;
    params.imgSize = (task == "detect") ? std::vector<int>{640, 640} : std::vector<int>{224, 224};
    params.rectConfidenceThreshold = 0.25;
    params.iouThreshold = 0.45;
    params.modelType = (task == "detect") ? YOLO_DETECT_V8 : YOLO_CLS;

#ifdef USE_CUDA
    params.cudaEnable = true;
#else
    params.cudaEnable = false;
#endif

    char* ret = yolo.CreateSession(params);
    if (ret != RET_OK) {
        std::cerr << "Failed to create session: " << ret << std::endl;
        return 1;
    }

    std::vector<std::string> classes = ReadClassNames(yamlPath);
    if (classes.empty()) {
        std::cerr << "Failed to read class names" << std::endl;
        return 1;
    }

    std::vector<DL_RESULT> results;
    ret = yolo.ProcessInput(inputPath, results);

    if (ret != RET_OK) {
        std::cerr << "Failed to process input: " << ret << std::endl;
        return 1;
    }

    cv::Mat img = cv::imread(inputPath);
    
    if (task == "detect") {
        VisualizeAndSaveDetection(img, results, classes, outputPath);
    } else {
        VisualizeAndSaveClassification(img, results, classes, outputPath);
    }

    return 0;
}