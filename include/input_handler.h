#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "inference.h"

namespace yolo {
    enum class InputType {
        IMAGE
        , VIDEO
        , STREAM
        , DIRECTORY
        , UNKNOWN
    };

    struct DetectionResult {
        cv::Mat frame;
        std::vector<DL_RESULT> detections;
    };

    class InputHandler {
        public:
            InputHandler(YOLOV8ONNX* detector, const std::string& outputDir = "" );
            ~InputHandler();

            void process(const std::string& source);
            void setOutputDirectory(const std::string& dir);
            void enableDisplay(bool enable) { showDisplay = enable; }

        private:
            InputType detectSourceType(const std::string& source);
            
            void processImage(const std::string& path);
            void processVideo(const std::string& path);
            void processStream(const std::string& url);
            void processDirectory(const std::string& path);

            void setupVideoWriter(const std::string& path, cv::VideoCapture& cap);


            void visualizeFrame(cv::Mat& frame, const std::vector<DL_RESULT>& results);
            void saveResult(const cv::Mat& frame, const std::string& orignalPath);

            YOLOV8ONNX* detector;

            std::string outputDir;

            bool showDisplay;

            cv::VideoWriter videoWriter;

            const std::vector<std::string> videoExtensions{".mp4", ".avi", ".mov", ".mkv"};
            const std::vector<std::string> imageExtensions{".jpg", ".jpeg", ".png", ".bmp"};

    };

}