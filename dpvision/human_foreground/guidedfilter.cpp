#include "guidedfilter.h"

void GuidedFilter::init_guide(cv::Mat& I, int r, double eps, int resize_rate) {
    //==== new impl =======
    resize_rate_ = resize_rate;
    I.convertTo(Ifr, CV_32FC3, 0.00392);
    resize(Ifr, If, Size(Ifr.cols / resize_rate, Ifr.rows / resize_rate));
    r /= resize_rate;
    r = 2 * r + 1;

    var_base = Mat(If.size(), CV_64FC(6));
    int size = If.cols * If.rows;
    blur(If, mean_base, Size(r, r));

    float* pI, * pM;
    double* pV;
    pI = If.ptr<float>(0);
    pM = mean_base.ptr<float>(0);
    pV = var_base.ptr<double>(0);
    for (int i = 0; i < size; ++i, pI += 3, pV += 6)
    {
        pV[0] = pI[0] * pI[0];
        pV[1] = pI[0] * pI[1];
        pV[2] = pI[0] * pI[2];
        pV[3] = pI[1] * pI[1];
        pV[4] = pI[1] * pI[2];
        pV[5] = pI[2] * pI[2];
    }
    blur(var_base, var_base, Size(r, r));
    pI = If.ptr<float>(0);
    pM = mean_base.ptr<float>(0);
    pV = var_base.ptr<double>(0);
    double tp0, tp1, tp2, tp3, tp4, tp5, det;
    for (int i = 0; i < size; ++i, pV += 6, pM += 3)
    {
        pV[0] = pV[0] - pM[0] * pM[0] + eps;
        pV[1] = pV[1] - pM[0] * pM[1];
        pV[2] = pV[2] - pM[0] * pM[2];
        pV[3] = pV[3] - pM[1] * pM[1] + eps;
        pV[4] = pV[4] - pM[1] * pM[2];
        pV[5] = pV[5] - pM[2] * pM[2] + eps;

        tp0 = pV[3] * pV[5] - pV[4] * pV[4];
        tp1 = pV[4] * pV[2] - pV[1] * pV[5];
        tp2 = pV[1] * pV[4] - pV[3] * pV[2];
        tp3 = pV[0] * pV[5] - pV[2] * pV[2];
        tp4 = pV[2] * pV[1] - pV[0] * pV[4];
        tp5 = pV[0] * pV[3] - pV[1] * pV[1];

        det = tp0 * pV[0] + tp1 * pV[1] + tp2 * pV[2];

        pV[0] = tp0 / det;
        pV[1] = tp1 / det;
        pV[2] = tp2 / det;
        pV[3] = tp3 / det;
        pV[4] = tp4 / det;
        pV[5] = tp5 / det;
    }
    //=====================

    this->r = r;
}

cv::Mat GuidedFilter::filter(cv::Mat& po) {
    Mat p;
    po.convertTo(pr, CV_32FC1, 0.00392);
    Mat result = Mat(pr.size(), CV_8UC1);
    //==== new impl =======
    resize(pr, p, Size(pr.cols / resize_rate_, pr.rows / resize_rate_));
    var_res = Mat(p.size(), CV_32FC(4));
    int size = p.cols * p.rows;

    blur(p, mean_res, Size(r, r));
    float* pI, * pMr, * pVr, * pM, * pP;
    double* pV;
    pP = p.ptr<float>(0);
    pI = If.ptr<float>(0);
    pMr = mean_res.ptr<float>(0);
    pVr = var_res.ptr<float>(0);
    pM = mean_base.ptr<float>(0);
    pV = var_base.ptr<double>(0);
    for (int i = 0; i < size; ++i, pP++, pVr += 4, pI += 3)
    {
        pVr[0] = pI[0] * pP[0];
        pVr[1] = pI[1] * pP[0];
        pVr[2] = pI[2] * pP[0];
    }
    blur(var_res, var_res, Size(r, r));
    pI = If.ptr<float>(0);
    pMr = mean_res.ptr<float>(0);
    pVr = var_res.ptr<float>(0);
    pM = mean_base.ptr<float>(0);
    pV = var_base.ptr<double>(0);
    double tp0, tp1, tp2;
    for (int i = 0; i < size; ++i, pMr++, pVr += 4, pM += 3, pV += 6)
    {
        tp0 = pVr[0] - pM[0] * pMr[0];
        tp1 = pVr[1] - pM[1] * pMr[0];
        tp2 = pVr[2] - pM[2] * pMr[0];

        pVr[0] = pV[0] * tp0 + pV[1] * tp1 + pV[2] * tp2;
        pVr[1] = pV[1] * tp0 + pV[3] * tp1 + pV[4] * tp2;
        pVr[2] = pV[2] * tp0 + pV[4] * tp1 + pV[5] * tp2;

        pVr[3] = pMr[0] - pVr[0] * pM[0] - pVr[1] * pM[1] - pVr[2] * pM[2];
    }

    blur(var_res, var_res, Size(r, r));

    resize(var_res, var_res, pr.size());


    Mat pro_msk;
    blur(pr, pro_msk, Size(5, 5));
    uchar* pR = result.ptr<uchar>(0);
    float* pT = pro_msk.ptr<float>(0);
    pP = pr.ptr<float>(0);
    pI = Ifr.ptr<float>(0);
    pVr = var_res.ptr<float>(0);

    size = pr.cols * pr.rows;
    for (int i = 0; i < size; ++i, pR += 1, pI += 3, pVr += 4, pT++, pP++)
    {
        float res = pVr[0] * pI[0] + pVr[1] * pI[1] + pVr[2] * pI[2] + pVr[3];
        float r = (pT[0] - 0.5) * 2.0;
        if (pT[0] > 0.5) {
            r = r * r;
            res = pP[0] * r + res * (1.0 - r);
        }
        else {
            float rate = 1 + r;
            res = res * rate * rate;
        }
        pR[0] = min(255.0f, max(0.0f, res * 255));
        // pR[0] = min(1.0f, max(0.0f, res));
        // pR[1] = min(1.0f, max(0.0f, res));
        // pR[2] = min(1.0f, max(0.0f, res));
        // pR[3] = min(1.0f, max(0.0f, res));
    }


    return result;

    //=====================
}
