#include "realsense_handler.h"
#include <iostream>

namespace yolo {
    RealSenseHandler::RealSenseHandler()
    : align_to_color(RS2_STREAM_COLOR)
    , depth_scale(0.001f)
    , filtering_enabled(false)
    , generate_pointcloud(false) {
        setupFilters();
    }

    RealSenseHandler::~RealSenseHandler() {
        stop();
    }

    void RealSenseHandler::setupFilters() {
        // Depth filer params
        spatial.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
        spatial.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5f);
        spatial.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20);
        
        temporal.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
        temporal.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20);
        
        dec.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
        
        thr.set_option(RS2_OPTION_MIN_DISTANCE, 0.15f);
        thr.set_option(RS2_OPTION_MAX_DISTANCE, 10.0f);
    }

    bool RealSenseHandler::initialize(int width, int height, int fps) {
        try {
            cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
            cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
            
            // Init pipe
            profile = pipe.start(cfg);

            // get intrinsics params
            auto depth_stream   = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            auto color_stream   = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

            color_intrinsics    = color_stream.get_intrinsics();
            depth_intrinsics    = depth_stream.get_intrinsics();

            // depth scale
            auto depth_sensor   = profile.get_device().first<rs2::depth_sensor>();
            depth_scale         = depth_sensor.get_depth_scale();

            return true;
        }
        catch(const rs2::error& e) {
            std::cerr << "RealSense error: " << e.what() << std::endl;
            return false;
        }
    }

    bool RealSenseHandler::getFrames(cv::Mat& color_frame, cv::Mat& depth_frame) {
        try {
            rs2::frameset frames    = pipe.wait_for_frames();
            frames                  = align_to_color.process(frames);
            
            // Get color frame
            rs2::frame color        = frames.get_color_frame();
            color_frame             = frame_to_mat(color);
            
            // Get depth frame
            rs2::frame depth        = frames.get_depth_frame();
            if (filtering_enabled) {
                applyDepthFilters(depth);
            }
            depth_frame             = frame_to_mat(depth);
            
            return true;
        }
        catch(const rs2::error& e) {
            std::cerr << "Frame acquisition error: " << e.what() << std::endl;
            return false;
        }
    }
    
}