#pragma once

#include "../base_face_detect.h"
#include <memory>
using std::shared_ptr;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT 
#endif

namespace DPFACE_MTCNN
{
    struct Bbox
    {
        float score;
        int x1;
        int y1;
        int x2;
        int y2;
        float area;
        float ppoint[10];
        float regreCoord[4];
    };

    class DLLEXPORT DPMTCNN : public DPFACE::CBaseFaceDetect
    {
    public:
        DPMTCNN();
        ~DPMTCNN();
        bool init(const string& config_file_path);
        bool detect(const cv::Mat& src, vector<cv::Rect2f>& faces, vector<float>& scores);
        bool detect(const cv::Mat& src, vector<DPFACE::DPFaceStruct>& faces);        
        bool detect(const cv::Mat& src, vector<DPFACE::DPFaceStruct>& faces, float factor, int stages);
        bool align(const cv::Mat& src, DPFACE::DPFaceStruct& face, int sidelength = 112);
    private:
        void generateBBox(cv::Mat& score, cv::Mat& location, vector<Bbox>& bboxs, float scale);
        void nms(vector<Bbox>& bboxs, const float overlap_threshold, string modelname = "Union");
        void refineAndSquareBbox(vector<Bbox>& bboxs, const int height, const int width);
        void proposalStage(const cv::Mat& src, vector<Bbox>& bboxs, float factor);
        void nextStage(const cv::Mat& src, vector<Bbox>& prev_bboxs, int stage);
    private:
        shared_ptr<void> pnet, rnet, onet;
        shared_ptr<void> aligner;
        vector<float> thrs;
        vector<float> nms_thr;

        int pnet_stride;
        int pnet_cell_size;

        std::vector<Bbox> first_bboxes, second_bboxes, third_bboxes;
    };
}