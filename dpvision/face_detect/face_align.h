#ifndef ARSENAL_HEADER_FILE
#define ARSENAL_HEADER_FILE

#include "base_face_detect.h"
#include <opencv2/opencv.hpp>

using DPFACE::DPFaceStruct;
using DPFACE::LANDMARK_5_LEN;

const int front_face_3D_pt_num = 13;
const int front_face_2D_pt_num = 13;

struct Facial_3D_pt
{
    float x;
    float y;
    float z;
};

struct Facial_2D_pt
{
    float x;
    float y;
};

const Facial_3D_pt face_3D_pt_mean_model_const[front_face_3D_pt_num] =
{
    -0.480398, 0.827507, -0.320129,
    -1.229336, 0.875213, -0.447955,
    0.480406, 0.827510, -0.320129,
    1.229335, 0.875212, -0.447955,
    0.000006, 0.825324, 0.120760,
    0.000008, -0.045678, 0.753794,
    0.000007, 0.374868, 0.415958,
    0.000007, -0.374110, 0.299188,
    -0.414896, -0.305268, 0.002681,
    0.414906, -0.305270, 0.002681,
    -0.692012, -0.950933, -0.183603,
    0.692025, -0.950928, -0.183603,
    0.000008, -0.703830, 0.308352
};

class CFaceAlign
{
public:
    CFaceAlign();
    ~CFaceAlign() {};

    void faceAlign5Points(DPFaceStruct& a_face,
        cv::Mat src,
        int crop_length);
    void faceAlign5PointsLS(DPFaceStruct& a_face,
        cv::Mat src,
        int crop_length);

private:
    void Projection_From_3D_To_2D();
    void Affine_transformation_2D(Facial_2D_pt face_key_2D_getpt[], unsigned char* get_norm_image, unsigned char* image, int ht, int wd);
    void ClaAffineTransfromCoeff_float(float* pt2_x, float* pt2_y, float* pt1_x, float* pt1_y, int npt, float& rot_s_x, float& rot_s_y, float& move_x, float& move_y);
    void ImageAffineTransform_Sam_Bilinear(float rot_s_x, float rot_s_y, float move_x, float move_y, unsigned char* image, int ht, int wd, unsigned char* ori_image, int oriht, int oriwd);
    bool MatrixTranspose(float* m1, int row1, int col1, float* m2);
    bool MatrixMulti(float* m1, int row1, int col1, float* m2, int row2, int col2, float* m3);
    bool MatrixInverse(float* m1, int row1, int col1);

private:

    Facial_3D_pt face_3D_pt_mean_model[front_face_3D_pt_num];
    Facial_2D_pt face_2D_pt_mean_model[front_face_2D_pt_num];
    Facial_2D_pt face_2D_pt_regression_mean_model[front_face_2D_pt_num];
    int norm_texture_wd_ht;
    int norm_regression_wd_ht;
    int affine_norm_wd_ht;
    int final_norm_wd_ht;
    float projection_focus_factor;
    float mean_texture_projection_move_z;
    float std_mean_focus;

public:
    bool CropImage_112x96(const cv::Mat& img, const float* facial5point, cv::Mat& crop);
    bool CropImage_112x112(const cv::Mat& img, const float* facial5point, cv::Mat& crop);
    bool CropImage_120x120(const cv::Mat& img, const float* facial5point, cv::Mat& crop);
    bool CropImage_140x140(const cv::Mat& img, const float* facial5point, cv::Mat& crop);
    bool CropImage_160x160(const cv::Mat& img, const float* facial5point, cv::Mat& crop);

private:

    void _findNonreflectiveSimilarity(int nPts, const float* uv, const float* xy, cv::Mat& transform);
    void _findSimilarity(int nPts, const float* uv, const float* xy, cv::Mat& transform);
};

#endif