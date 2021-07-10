#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <memory>

using std::string;
using std::vector;
using std::shared_ptr;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT 
#endif

struct NetworkParam
{
    cv::Size dst_sz;
    cv::Size crop_sz;
};

class DLLEXPORT DPCNN_CPU
{
public:
    DPCNN_CPU();
    ~DPCNN_CPU();

    bool newCNN(const string& model_file,
        const string& mean_value,
        const string& scale_value);

    bool newCNN(const unsigned char model_bin[],
        const int model_size,
        const string& mean_value,
        const string& scale_value);

    bool newCNN(const string& config_file_path);

    vector<float> predict(const cv::Mat& img, const int& num_thread = 0, const cv::Size& force_size = cv::Size(0, 0));
    vector<cv::Mat> predictMap(const cv::Mat& img, const int& num_thread = 0, const cv::Size& force_size = cv::Size(0, 0));

    NetworkParam getNetParam();

protected:
    void release();
    shared_ptr<void> dpcnn;
    NetworkParam net_param;
};