#pragma once
#include <opencv2/opencv.hpp>

#include <memory>
#include <vector>
#include <string>

using std::shared_ptr;
using std::string;
using std::vector;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

namespace DP_HUMAN_FOREGROUND
{
    class DLLEXPORT CHumanForeground
    {
    public:
        CHumanForeground();
        ~CHumanForeground();

        /**
         @param config_file：配置文件，即fbs_config.txt所在位置。
         @param dst_sz：输出图片的大小
         @param back：用来替换的背景图片
         @return：初始化是否成功
        */
        bool init(const cv::Size& dst_sz, const cv::Mat& back, const std::string& license_path);

        /**
         @param src：待分割的图片
         @param ret：替换背景之后的图片
         @param num_thread：用多少个线程来并行计算，默认为0使用最多线程
        */
        cv::Mat human_foreground(const cv::Mat& src, const int& num_thread = 0);

    protected:
        void release();
        std::shared_ptr<void> cnn;
        std::shared_ptr<void> guided_filter;

        int r;
        double eps;
        cv::Size dst_size;

        cv::Mat src_process, mask, mask_post, foreground, back_f, back_u, heatmap;
        cv::Mat alpha_32fc3;
        vector<cv::Mat> pre_img;
        vector<cv::Mat> pre_mask;
        vector<cv::Mat> splits;
    };
} // namespace DP_HUMAN_FOREGROUND