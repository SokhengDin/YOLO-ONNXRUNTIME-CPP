#include "input_handler.h"
#include <algorithm>

namespace yolo {
    
    InputHandler::InputHandler(YOLOV8ONNX* detector, const std::string& outputDir)
    : detector(detector), outputDir(outputDir), showDisplay(true) {
        if (!outputDir.empty() && !std::filesystem::exists(outputDir)) {
            std::filesystem::create_directories(outputDir);
        }
    }

    InputHandler::~InputHandler() {
        if (videoWriter.isOpened()) {
            videoWriter.release();
        }

        cv::destroyAllWindows();
    }

    void InputHandler::process(const std::string& source) {
        switch (detectSourceType(source)) {
            case InputType::IMAGE:          processImage(source); break;
            case InputType::VIDEO:      processVideo(source); break;
            case InputType::STREAM:     processStream(source); break;
            case InputType::DIRECTORY:  processDirectory(source); break;
            default: std::cerr << "Unsupported input type" << std::endl;
        }
    }

    InputType InputHandler::detectSourceType(const std::string& source) {
        if (source.substr(0, 7) == "rtsp://" || 
            source.substr(0, 7) == "http://") {
            return InputType::STREAM;
        }

        if (std::filesystem::is_directory(source)) {
            return InputType::DIRECTORY;
        }

        std::string ext = std::filesystem::path(source).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (std::find(videoExtensions.begin(), videoExtensions.end(), ext) != videoExtensions.end()) {
            return InputType::VIDEO;
        }
        if (std::find(imageExtensions.begin(), imageExtensions.end(), ext) != imageExtensions.end()) {
            return InputType::IMAGE;
        }

        return InputType::UNKNOWN;
    }

    void InputHandler::processImage(const std::string& path) {
        cv::Mat frame   = cv::imread(path);

        if (frame.empty()) return;

        std::vector<DL_RESULT> results;

        detector->RunSession(frame, results);

        visualizeFrame(frame, results);
        if (!outputDir.empty()) {
            saveResult(frame, path);
        }

        if (showDisplay) {
            cv::imshow("Detection", frame);
            cv::waitKey(0);
        }
    }

    void InputHandler::processVideo(const std::string& path) {
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) return;

        setupVideoWriter(path, cap);

        cv::Mat frame;
        while (cap.read(frame)) {
            std::vector<DL_RESULT> results;
            detector->RunSession(frame, results);
            
            visualizeFrame(frame, results);
            
            if (videoWriter.isOpened()) {
                videoWriter.write(frame);
            }

            if (showDisplay) {
                cv::imshow("Detection", frame);
                if (cv::waitKey(1) == 27) break;
            }
        }

        cap.release();
    }

    void InputHandler::processStream(const std::string& url) {
        cv::VideoCapture cap(url);
        if (!cap.isOpened()) return;

        cap.set(cv::CAP_PROP_BUFFERSIZE, 3);

        cv::Mat frame;
        while (cap.read(frame)) {
            std::vector<DL_RESULT> results;
            detector->RunSession(frame, results);
            
            visualizeFrame(frame, results);

            if (showDisplay) {
                cv::imshow("Stream", frame);
                if (cv::waitKey(1) == 27) break;
            }
        }

        cap.release();
    }

    void InputHandler::processDirectory(const std::string& path) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (detectSourceType(entry.path().string()) == InputType::IMAGE) {
                processImage(entry.path().string());
            }
        }
    }

    void InputHandler::visualizeFrame(cv::Mat& frame, const std::vector<DL_RESULT>& results) {
        for (const auto& result : results) {
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
            cv::rectangle(frame, result.box, color, 2);

            std::string label = detector->classes[result.classId] + 
                            " " + std::to_string(static_cast<int>(result.confidence * 100)) + "%";

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.75, 2, &baseline);
            cv::rectangle(frame,
                cv::Point(result.box.x, result.box.y - textSize.height - 5),
                cv::Point(result.box.x + textSize.width, result.box.y),
                color, cv::FILLED);

            cv::putText(frame, label,
                cv::Point(result.box.x, result.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 2);
        }
    }

    void InputHandler::setupVideoWriter(const std::string& path, cv::VideoCapture& cap) {
        if (outputDir.empty()) return;
        
        std::string fileName = "result_" + std::filesystem::path(path).filename().string();
        std::string outputPath = (std::filesystem::path(outputDir) / fileName).string();
        
        int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frameSize(
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
        
        videoWriter.open(outputPath, fourcc, fps, frameSize);
    }

    void InputHandler::saveResult(const cv::Mat& frame, const std::string& originalPath) {
        if (outputDir.empty()) return;
        
        std::string fileName = "result_" + std::filesystem::path(originalPath).filename().string();
        std::string outputPath = (std::filesystem::path(outputDir) / fileName).string();
        cv::imwrite(outputPath, frame);
    }

    void InputHandler::setOutputDirectory(const std::string& dir) {
        outputDir = dir;
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }
    }

}