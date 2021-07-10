#include "human_foreground.h"
#include "guidedfilter.h"
#include "dpcnn.h"
#include "util.h"
#include "human_parsing.h"

#ifdef _DEBUG
#pragma comment(lib, "dpcnnd.lib")
#pragma comment(lib, "opencv_world346d.lib")
#else
#pragma comment(lib,"dpcnn.lib")
#pragma comment(lib, "opencv_world346.lib")
#endif

namespace DP_HUMAN_FOREGROUND
{
    CHumanForeground::CHumanForeground()
    {
        cnn = nullptr;
        guided_filter = nullptr;
    }

    CHumanForeground::~CHumanForeground()
    {
        release();
    }

    bool CHumanForeground::init(const cv::Size& dst_sz, const cv::Mat& back, const std::string& license_path)
    {
        release();
        bool ret = true;
        if (!verifyLicense(license_path))
            return false;
        cnn = std::static_pointer_cast<void>(std::shared_ptr<DPCNN_CPU>(new DPCNN_CPU));
        shared_ptr<DPCNN_CPU> cnn_ptr = std::static_pointer_cast<DPCNN_CPU>(cnn);
        ret = ret && cnn_ptr->newCNN(human_parsing_mnn, 1653108, "104.008,116.669,122.675", "0.00390625,0.00390625,0.00390625");

        r = 15;
        eps = 1e-3;
        guided_filter = std::static_pointer_cast<void>(std::shared_ptr<GuidedFilter>(new GuidedFilter));

        dst_size = dst_sz;

        back.copyTo(back_u);
        resize(back_u, back_u, dst_size);
        back_u.convertTo(back_u, CV_32FC3);

        alpha_32fc3 = cv::Mat(dst_size, CV_32FC3);
        return ret;
    }

    cv::Mat CHumanForeground::human_foreground(const cv::Mat& src, const int& num_thread)
    {
        alpha_32fc3.setTo(0);
        shared_ptr<DPCNN_CPU> cnn_ptr = std::static_pointer_cast<DPCNN_CPU>(cnn);
        heatmap = cnn_ptr->predictMap(src, num_thread)[0];
        cv::split(heatmap, splits);
        mask = splits[1] * 255;
        cv::resize(mask, mask, dst_size);
        cv::resize(src, src_process, mask.size());

        shared_ptr<GuidedFilter> guided_filter_ptr = std::static_pointer_cast<GuidedFilter>(guided_filter);
        guided_filter_ptr->init_guide(src_process, r, eps, 4);
        mask_post = guided_filter_ptr->filter(mask);

        if (pre_mask.size() < 3)
        {
            pre_img.push_back(src_process);
            pre_mask.push_back(mask_post);
        }
        else
        {
            pre_img.push_back(src_process);
            pre_mask.push_back(mask_post);
            for (int i = 0; i < src_process.rows; ++i)
            {
                for (int j = 0; j < src_process.cols; ++j)
                {
                    int dis0 = pre_img[3].at<uchar>(i, j) - pre_img[0].at<uchar>(i, j);
                    int dis1 = pre_img[3].at<uchar>(i, j) - pre_img[1].at<uchar>(i, j);
                    int dis2 = pre_img[3].at<uchar>(i, j) - pre_img[2].at<uchar>(i, j);
                    float par0 = exp(-1 * dis0 * dis0 * 0.1);
                    float par1 = exp(-1 * dis1 * dis1 * 0.1);
                    float par2 = exp(-1 * dis2 * dis2 * 0.1);
                    float dst_value = (pre_mask[3].at<uchar>(i, j) + pre_mask[2].at<uchar>(i, j) + pre_mask[1].at<uchar>(i, j)) * 0.0013;
                    cv::Vec3f& p = alpha_32fc3.at<cv::Vec3f>(i, j);
                    p[0] = dst_value;
                    p[1] = dst_value;
                    p[2] = dst_value;
                }
            }
            pre_img.erase(pre_img.begin());
            pre_mask.erase(pre_mask.begin());
        }

        src_process.convertTo(foreground, CV_32FC3);
        cv::multiply(alpha_32fc3, foreground, foreground);

        // Multiply the background with ( 1 - alpha )
        cv::multiply(Scalar::all(1.0) - alpha_32fc3, back_u, back_f);

        // Add the masked foreground and background.
        cv::Mat ret = Mat::zeros(foreground.size(), foreground.type());
        cv::add(foreground, back_f, ret);
        ret.convertTo(ret, CV_8UC3);
        return ret;
    }
    void CHumanForeground::release()
    {
        if (cnn != nullptr)
        {
            shared_ptr<DPCNN_CPU> cnn_ptr = std::static_pointer_cast<DPCNN_CPU>(cnn);
            cnn_ptr.reset();
        }
        if (guided_filter != nullptr)
        {
            shared_ptr<GuidedFilter> guided_filter_ptr = std::static_pointer_cast<GuidedFilter>(guided_filter);
            guided_filter_ptr.reset();
        }
    }
} // namespace DP_HUMAN_FOREGROUND