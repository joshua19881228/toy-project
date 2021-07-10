#pragma once

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <opencv2/opencv.hpp>
#include <string>

class MNNWraper
{
public:
    MNNWraper();
    ~MNNWraper();

    bool init(const std::string& param_path,
        const std::vector<float>& means,
        const std::vector<float>& scales);

    bool init_mem(const unsigned char* model_bin,
        const int model_size,
        const std::vector<float>& means,
        const std::vector<float>& scales);

    std::vector<cv::Mat> predictMat(const cv::Mat& src, const int& num_thread = 0, const cv::Size& force_size = cv::Size(0, 0));
    std::vector<float> predictFeat(const cv::Mat& src, const int& num_thread = 0, const cv::Size& force_size = cv::Size(0, 0));

public:
    cv::Size dst_size;
    cv::Size crop_size;
private:

    void release(bool rls_net = false);
    bool checkParams();

    std::vector<float> means;
    std::vector<float> scales;

    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session;
};