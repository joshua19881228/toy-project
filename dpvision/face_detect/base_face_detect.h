#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#ifdef _DEBUG
#pragma comment(lib,"dpcnnd.lib")
#pragma comment(lib,"opencv_world346d.lib")
#else
#pragma comment(lib,"dpcnn.lib")
#pragma comment(lib,"opencv_world346.lib")
#endif

using std::string;
using std::vector;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT 
#endif

namespace DPFACE
{
    const int LANDMARK_5_LEN = 5;
    const int LANDMARK_51_LEN = 51;
    const int FACE_FEAT_LEN = 256;

    struct DPFaceStruct
    {
        int id;
        float feats[FACE_FEAT_LEN];
        float landmarks_5[LANDMARK_5_LEN][2];
        float landmarks_51[LANDMARK_51_LEN][2];
        float axis_3d[3];
        float score;
        bool is_detected;
        cv::Mat aligned;
        float x, y, w, h;

        DPFaceStruct& operator=(DPFaceStruct& value)
        {
            id = value.id;
            x = value.x;
            y = value.y;
            w = value.w;
            h = value.h;
            memcpy(feats, value.feats, FACE_FEAT_LEN * sizeof(float));
            memcpy(landmarks_5, value.landmarks_5, LANDMARK_5_LEN * 2 * sizeof(float));
            memcpy(landmarks_51, value.landmarks_51, LANDMARK_51_LEN * 2 * sizeof(float));
            memcpy(axis_3d, value.axis_3d, 3 * sizeof(float));
            value.aligned.copyTo(aligned);
            return *this;
        }
    };

    class DLLEXPORT CBaseFaceDetect
    {
    public:
        virtual bool init(const string& config_file) = 0;
        virtual bool detect(const cv::Mat& src, vector<DPFaceStruct>& faces) = 0;
        virtual bool align(const cv::Mat& src, DPFaceStruct& face, int sidelength) = 0;
    protected:
        int min_size;
        int max_size;
    };
}