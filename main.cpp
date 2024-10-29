#include "input_handler.h"

int ReadCocoYaml(YOLOV8ONNX*& p) {
        // Open the YAML file
        std::ifstream file("coco.yaml");
        if (!file.is_open())
        {
            std::cerr << "Failed to open file" << std::endl;
            return 1;
        }

        std::string line;
        std::vector<std::string> lines;
        while (std::getline(file, line))
        {
            lines.push_back(line);
        }
        std::size_t start = 0;
        std::size_t end = 0;
        for (std::size_t i = 0; i < lines.size(); i++)
        {
            if (lines[i].find("names:") != std::string::npos)
            {
                start = i + 1;
            }
            else if (start > 0 && lines[i].find(':') == std::string::npos)
            {
                end = i;
                break;
            }
        }

        // Extract the names
        std::vector<std::string> names;
        for (std::size_t i = start; i < end; i++)
        {
            std::stringstream ss(lines[i]);
            std::string name;
            std::getline(ss, name, ':'); 
            std::getline(ss, name);
            names.push_back(name);
        }

        p->classes = names;
        return 0;
    }

void DetectTest() {
    YOLOV8ONNX* detector = new YOLOV8ONNX;
    ReadCocoYaml(detector);

    DL_INIT_PARAM params;
    params.rectConfidenceThreshold  = 0.25;
    params.iouThreshold             = 0.45;
    params.modelPath                = "yolov8n.onnx";
    params.imgSize                  = { 640, 640 };
    params.modelType                = YOLO_DETECT_V8;
    
#ifdef USE_CUDA
    params.cudaEnable = true;
#endif

    if (detector->CreateSession(params) != RET_OK) {
        delete detector;
        return;
    }

    yolo::InputHandler handler(detector, "output");

    // handler.process("test.jpg");                    // Single image
    // handler.process("video.mp4");                   // Video file
    // handler.process("rtsp://camera_url");           // RTSP stream
    handler.process("images");                      // Directory of images

    delete detector;
}

int main() {
    DetectTest();
    return 0;
}