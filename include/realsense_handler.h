#pragma once

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include "inference.h"
#include <memory>
#include <vector>

namespace yolo {
    struct DetectionWithDepth
    {
        DL_RESULT detection;
        float distance;
        cv::Point3f position; // 3D position (x,y,z) in camera frame
    };

    class RealSenseHandler {
        public:
            RealSenseHandler();
            ~RealSenseHandler();

            // initliazer
            bool initialize(int width = 640, int height = 480, int fps = 30);

            // Get  next frame with depth data, this is video
            bool getFrames(cv::Mat& color_frame, cv::Mat& depth_frame);

            // Process detections with depth information
            std::vector<DetectionWithDepth> processDetections(
                const std::vector<DL_RESULT>& detections,
                const cv::Mat& depth_frame
            )

            rs2_intrinsics getCameraIntrinsics() const { return color_intrinsics; }

            float getDepthScale() const { return depth_scale; }
            cv::Point3f deprojectPixelToPoint(const cv::Point2f& pixel, float depth) const;

            void start();
            void stop();

            void enablePointCloud(bool enable) { generate_pointcloud = enable; }
            void setDepthFilter(bool enable);
            void setEmitterState(bool enable);
        
        private:

            // RS pipline
            rs2::pipeline pipe;
            rs2::config cfg;
            rs2::pipeline_profile profile;

            // Camera params
            rs2_intrinsics color_intrinsics;
            rs2_intrinsics depth_intrinsics;
            rs2::align align_to_color;
            float depth_scale;

            // Processes blocks
            rs2::spatial_filter spatial;
            rs2::temporal_filter temporal;
            rs2::decimation_filter dec;
            rs2::threshold_filter thr;
            rs2::pointcloud pc;

            bool filtering_enabled;
            bool generate_pointcloud;

            // Internal methods
            void setupFilters();
            cv::Mat frame_to_mat(const rs2::frame& frame);
            float getAverageDepth(const cv::Rect& bbox, const cv::Mat& depth_frame);
            void applyDepthFilters(rs2::frame& depth_frame);
    }
    
}