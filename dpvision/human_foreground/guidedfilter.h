#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;

class GuidedFilter
{
public:
    GuidedFilter() {};
    ~GuidedFilter() {};

    void init_guide(Mat& I, int r, double eps, int resize_rate = 1);
    Mat filter(Mat& po);

private:
    std::vector<Mat> Ichannels;
    int r;
    double eps;
    int resize_rate_;

    Mat If, mean_base, var_base;
    Mat mean_res, var_res;
    Mat Ifr, pr;

};

#endif
