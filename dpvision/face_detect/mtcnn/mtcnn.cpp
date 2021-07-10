#include "mtcnn.h"
#include "dpcnn.h"
#include "../face_align.h"
#include "det1.mem.28052.h"
#include "det2.mem.403416.h"
#include "det3.mem.398620.h"
#include "util.h"

namespace DPFACE_MTCNN
{
    vector<cv::Mat> splitCVMat(const cv::Mat& src)
    {
        vector<cv::Mat> splits;
        cv::split(src, splits);
        return splits;
    }

    bool cmpScore(Bbox lsh, Bbox rsh) {
        if (lsh.score < rsh.score)
            return true;
        else
            return false;
    }

    bool cmpArea(Bbox lsh, Bbox rsh) {
        if (lsh.area < rsh.area)
            return false;
        else
            return true;
    }

    DPMTCNN::DPMTCNN()
    {

    }

    DPMTCNN::~DPMTCNN()
    {

    }

    bool DPMTCNN::init(const string& config_file_path)
    {
        bool ret = true;

        //check config file
        std::map<std::string, std::string> configs;
        if (loadConfigFile(config_file_path, configs) != 0)
        {
            LOGE("load mtcnn config file failed");
            return false;
        }
        if (configs.find("min_size") == configs.end() ||
            configs.find("max_size") == configs.end() ||
            configs.find("thrs") == configs.end() ||
            configs.find("nms_thr") == configs.end() ||
            configs.find("stride") == configs.end() ||
            configs.find("cellsize") == configs.end())
        {
            LOGE("mtcnn config file corrupted in [%s]", config_file_path.c_str());
            LOGE("It should include\n min_size\n max_size\n stage1\n \
                                  stage2\n stage3\n thrs\n nms_thr\n stride\n cellsize\n");
            return false;
        }

        //load configs into variables
        //min max face size
        min_size = std::atoi(configs["min_size"].c_str());
        max_size = std::atoi(configs["max_size"].c_str());
        //pnet, rnet, onet
        pnet = std::static_pointer_cast<void>(std::shared_ptr<DPCNN_CPU>(new DPCNN_CPU));
        rnet = std::static_pointer_cast<void>(std::shared_ptr<DPCNN_CPU>(new DPCNN_CPU));
        onet = std::static_pointer_cast<void>(std::shared_ptr<DPCNN_CPU>(new DPCNN_CPU));

        shared_ptr<DPCNN_CPU> pnet_ptr = std::static_pointer_cast<DPCNN_CPU>(pnet);
        shared_ptr<DPCNN_CPU> rnet_ptr = std::static_pointer_cast<DPCNN_CPU>(rnet);
        shared_ptr<DPCNN_CPU> onet_ptr = std::static_pointer_cast<DPCNN_CPU>(onet);
        //ret = ret && pnet_ptr->newCNN(configs["stage1"]);
        //ret = ret && rnet_ptr->newCNN(configs["stage2"]);
        //ret = ret && onet_ptr->newCNN(configs["stage3"]);
        ret = ret && pnet_ptr->newCNN(det1_mnn,
            28052,
            "127.5,127.5,127.5",
            "0.0078125,0.0078125,0.0078125");
        ret = ret && rnet_ptr->newCNN(det2_mnn,
            403416,
            "127.5,127.5,127.5",
            "0.0078125,0.0078125,0.0078125");
        ret = ret && onet_ptr->newCNN(det3_mnn,
            398620,
            "127.5,127.5,127.5",
            "0.0078125,0.0078125,0.0078125");
        //thresholds of stages
        string item;
        std::stringstream thrs_ss(configs["thrs"]);
        while (getline(thrs_ss, item, ',')) {
            thrs.push_back(std::atof(item.c_str()));
        }
        if (thrs.size() != 3)
        {
            LOGE("THREE thresholds of stages");
            ret = ret && false;
        }
        //thresholds of nms
        std::stringstream nms_thr_ss(configs["nms_thr"]);
        while (getline(nms_thr_ss, item, ',')) {
            nms_thr.push_back(std::atof(item.c_str()));
        }
        if (nms_thr.size() != 3)
        {
            LOGE("THREE thresholds of nms");
            ret = ret && false;
        }
        //params for pnet
        pnet_stride = std::atoi(configs["stride"].c_str());
        pnet_cell_size = std::atoi(configs["cellsize"].c_str());
        //face aligner
        aligner.reset(new CFaceAlign);

        return ret;
    }

    bool DPMTCNN::detect(const cv::Mat& src, vector<DPFaceStruct>& faces)
    {
        bool ret = true;
        ret = ret && detect(src, faces, 0.709, 2);
        return ret;
    }

    bool DPMTCNN::detect(const cv::Mat& src, vector<cv::Rect2f>& faces, vector<float>& scores)
    {
        bool ret = true;
        faces.clear();
        scores.clear();
        vector<DPFaceStruct> faces_tmp;
        ret = ret && detect(src, faces_tmp, 0.709, 2);
        if (ret)
        {
            for (int i = 0; i < faces_tmp.size(); ++i)
            {
                faces.push_back(cv::Rect2f(faces_tmp[i].x, faces_tmp[i].y, faces_tmp[i].w, faces_tmp[i].h));
                scores.push_back(faces_tmp[i].score);
            }
        }
        return ret;
    }

    void drawBox(cv::Mat& src, const vector<Bbox>& boxes)
    {
        for (int i = 0; i < boxes.size(); ++i)
        {
            cv::rectangle(src, cv::Rect(cv::Point(boxes[i].x1, boxes[i].y1), cv::Point(boxes[i].x2, boxes[i].y2)), cv::Scalar(255, 255, 255));
        }
        cv::imshow("drawBox", src);
        cv::waitKey(0);
    }

    bool DPMTCNN::detect(const cv::Mat& src, vector<DPFACE::DPFaceStruct>& faces, float factor, int stages /* = 2 */)
    {
        bool ret = true;
        faces.clear();
        vector<Bbox> final_ret;
        if (stages >= 0)
        {
            //double t = (double)cv::getTickCount();
            proposalStage(src, first_bboxes, factor);
            final_ret = first_bboxes;
            //std::cout << "pnet time cost: " << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s" << std::endl<<final_ret.size()<<std::endl;
#ifdef _DEBUG
            LOGI("first stage");
            cv::Mat show = src.clone();
            drawBox(show, final_ret);
#endif
        }
        if (stages >= 1 && first_bboxes.size() > 0)
        {
            //double t = (double)cv::getTickCount();
            nextStage(src, first_bboxes, 1);
            final_ret = second_bboxes;
            //std::cout << "rnet time cost: " << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s" << std::endl << final_ret.size() << std::endl;
#ifdef _DEBUG
            LOGI("second stage");
            cv::Mat show = src.clone();
            drawBox(show, final_ret);
#endif
        }
        if (stages >= 2 && second_bboxes.size() > 0)
        {
            //double t = (double)cv::getTickCount();
            nextStage(src, second_bboxes, 2);
            final_ret = third_bboxes;
            //std::cout << "onet time cost: " << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s" << std::endl << final_ret.size() << std::endl;
#ifdef _DEBUG
            LOGI("third stage");
            cv::Mat show = src.clone();
            drawBox(show, final_ret);
#endif
        }

        for (int i = 0; i < final_ret.size(); ++i)
        {
            DPFACE::DPFaceStruct aface;
            aface.x = (float)final_ret[i].x1 / src.cols;
            aface.y = (float)final_ret[i].y1 / src.rows;
            aface.w = (float)(final_ret[i].x2 - final_ret[i].x1 + 1) / src.cols;
            aface.h = (float)(final_ret[i].y2 - final_ret[i].y1 + 1) / src.rows;
            aface.id = i;
            aface.score = final_ret[i].score;
            for (int l = 0; l < 5; ++l)
            {
                aface.landmarks_5[l][0] = final_ret[i].ppoint[l] / src.cols;
                aface.landmarks_5[l][1] = final_ret[i].ppoint[l + 5] / src.rows;
            }
            faces.push_back(aface);
        }
        return ret;
    }

    bool DPMTCNN::align(const cv::Mat& src, DPFaceStruct& face, int sidelength)
    {
        shared_ptr<CFaceAlign> aligner_ptr = std::static_pointer_cast<CFaceAlign>(aligner);
        aligner_ptr->faceAlign5PointsLS(face, src, sidelength);
        return true;
    }

    void DPMTCNN::generateBBox(cv::Mat& score, cv::Mat& location, vector<Bbox>& bboxs, float scale)
    {
        double t = (double)cv::getTickCount();
        vector<cv::Mat> score_splits = splitCVMat(score);
        vector<cv::Mat> loc_splits = splitCVMat(location);
        int width = score_splits[1].cols;
        int height = score_splits[1].rows;
        Bbox bbox;
        float inv_scale = 1.0f / scale;
        float pred_score;

        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                pred_score = score_splits[1].at<float>(row * width + col);
                if (pred_score > thrs[0]) {
                    bbox.score = pred_score;//记录得分
                    bbox.x1 = round((pnet_stride * col + 1) * inv_scale);//12*12的滑框，换算到原始图像上的坐标
                    bbox.y1 = round((pnet_stride * row + 1) * inv_scale);
                    bbox.x2 = round((pnet_stride * col + pnet_cell_size) * inv_scale);
                    bbox.y2 = round((pnet_stride * row + pnet_cell_size) * inv_scale);
                    bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
                    for (int channel = 0; channel < 4; channel++)
                        bbox.regreCoord[channel] = loc_splits[channel].at<float>(0);//人脸框的坐标相关值
                    bboxs.push_back(bbox);
                }
            }
        }
    }

    void DPMTCNN::nms(vector<Bbox>& bboxs, const float overlap_threshold, string modelname /* = "Union" */)
    {
        if (bboxs.empty()) {
            return;
        }
        sort(bboxs.begin(), bboxs.end(), cmpScore);
        float IOU = 0;
        float maxX = 0;
        float maxY = 0;
        float minX = 0;
        float minY = 0;
        std::vector<int> vPick;
        int nPick = 0;
        std::multimap<float, int> vScores;
        const int num_boxes = bboxs.size();
        vPick.resize(num_boxes);
        for (int i = 0; i < num_boxes; ++i) {
            vScores.insert(std::pair<float, int>(bboxs[i].score, i));
        }
        while (vScores.size() > 0) {
            int last = vScores.rbegin()->second;
            vPick[nPick] = last;
            nPick += 1;
            for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
                int it_idx = it->second;
                maxX = std::max(bboxs.at(it_idx).x1, bboxs.at(last).x1);
                maxY = std::max(bboxs.at(it_idx).y1, bboxs.at(last).y1);
                minX = std::min(bboxs.at(it_idx).x2, bboxs.at(last).x2);
                minY = std::min(bboxs.at(it_idx).y2, bboxs.at(last).y2);
                //maxX1 and maxY1 reuse 
                maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
                maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if (!modelname.compare("Union"))
                    IOU = IOU / (bboxs.at(it_idx).area + bboxs.at(last).area - IOU);
                else if (!modelname.compare("Min")) {
                    IOU = IOU / ((bboxs.at(it_idx).area < bboxs.at(last).area) ? bboxs.at(it_idx).area : bboxs.at(last).area);
                }
                if (IOU > overlap_threshold) {
                    it = vScores.erase(it);
                }
                else {
                    it++;
                }
            }
        }

        vPick.resize(nPick);
        std::vector<Bbox> tmp_;
        tmp_.resize(nPick);
        for (int i = 0; i < nPick; i++) {
            tmp_[i] = bboxs[vPick[i]];
        }
        bboxs = tmp_;
    }

    void DPMTCNN::refineAndSquareBbox(vector<Bbox>& bboxs, const int height, const int width)
    {
        if (bboxs.empty()) {
            return;
        }
        float bbw = 0, bbh = 0, maxSide = 0;
        float h = 0, w = 0;
        float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        for (vector<Bbox>::iterator it = bboxs.begin(); it != bboxs.end(); it++)
        {
            bbw = (*it).x2 - (*it).x1 + 1;//滑框的宽高计算
            bbh = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[0] * bbw;//人脸框的位置坐标计算
            y1 = (*it).y1 + (*it).regreCoord[1] * bbh;
            x2 = (*it).x2 + (*it).regreCoord[2] * bbw;
            y2 = (*it).y2 + (*it).regreCoord[3] * bbh;

            w = x2 - x1 + 1;//人脸框宽高
            h = y2 - y1 + 1;

            maxSide = (h > w) ? h : w;
            x1 = x1 + w * 0.5 - maxSide * 0.5;
            y1 = y1 + h * 0.5 - maxSide * 0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if ((*it).x1 < 0)(*it).x1 = 0;
            if ((*it).y1 < 0)(*it).y1 = 0;
            if ((*it).x2 > width)(*it).x2 = width - 1;
            if ((*it).y2 > height)(*it).y2 = height - 1;

            it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
        }
    }

    void DPMTCNN::proposalStage(const cv::Mat& src, vector<Bbox>& bboxs, float factor)
    {
        bboxs.clear();
        shared_ptr<DPCNN_CPU> net = std::static_pointer_cast<DPCNN_CPU>(pnet);

        int width = src.cols;
        int height = src.rows;
        float multiplier = (float)pnet_cell_size / min_size;
        float dst_scale = (float)pnet_cell_size / max_size;
        float min_wh = (width < height ? width : height) * multiplier;

        vector<float> scales;
        while (min_wh >= pnet_cell_size && multiplier > dst_scale)
        {
            scales.push_back(multiplier);
            min_wh *= factor;
            multiplier *= factor;
        }

        double t = (double)cv::getTickCount();
        for (int i = 0; i < scales.size(); ++i)
        {
            vector<Bbox> stage_boxs;
            cv::Mat scale_img;
            int hs = (int)ceil(height * scales[i]);
            int ws = (int)ceil(width * scales[i]);
            cv::resize(src, scale_img, cv::Size(ws, hs));
            double t = (double)cv::getTickCount();
            //cv::cvtColor(scale_img, scale_img, CV_BGR2RGB);
            vector<cv::Mat> stage_ret = net->predictMap(scale_img, 0, scale_img.size());
            generateBBox(stage_ret[1], stage_ret[0], stage_boxs, scales[i]);
            nms(stage_boxs, nms_thr[0]);
            bboxs.insert(bboxs.end(), stage_boxs.begin(), stage_boxs.end());
        }
        if (bboxs.size() < 1)
            return;
        nms(bboxs, nms_thr[0]);//主会场擂台赛
        refineAndSquareBbox(bboxs, height, width);
    }

    void DPMTCNN::nextStage(const cv::Mat& src, vector<Bbox>& prev_bboxs, int stage)
    {
        shared_ptr<DPCNN_CPU> net;
        vector<Bbox>* bbox;
        vector<int> idx;
        if (stage == 1)
        {
            net = std::static_pointer_cast<DPCNN_CPU>(rnet);
            bbox = &second_bboxes;
            idx.push_back(1);
            idx.push_back(0);
        }
        else if (stage == 2)
        {
            net = std::static_pointer_cast<DPCNN_CPU>(onet);
            bbox = &third_bboxes;
            idx.push_back(2);
            idx.push_back(0);
            idx.push_back(1);
        }
        else
        {
            LOGE("NOT defined yet in MTCNN");
            return;
        }

        bbox->clear();
        for (vector<Bbox>::iterator it = prev_bboxs.begin(); it != prev_bboxs.end(); it++)
        {
            cv::Mat roi = src(cv::Rect(cv::Point(std::max((int)(*it).x1, 0), std::max((int)(*it).y1, 0)),
                cv::Point(std::min((int)(*it).x2, src.cols - 1), std::min((int)(*it).y2, src.rows - 1)))).clone();
            cv::resize(roi, roi, net->getNetParam().dst_sz);
            //cv::cvtColor(roi, roi, CV_BGR2RGB);
            vector<cv::Mat> stage_ret = net->predictMap(roi);
            vector<cv::Mat> loc_ret = splitCVMat(stage_ret[idx[1]]);
            vector<cv::Mat> score_ret = splitCVMat(stage_ret[idx[0]]);

            float score = score_ret[1].at<float>(0);
            if (score > thrs[stage])
            {
                for (int channel = 0; channel < 4; channel++)
                    (it->regreCoord)[channel] = loc_ret[channel].at<float>(0);//*(bbox.data+channel*bbox.cstep);
                it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
                it->score = score;//*(score.data+score.cstep);

                if (stage == 2)
                {
                    vector<cv::Mat> landmark_splits = splitCVMat(stage_ret[idx[2]]);
                    for (int num = 0; num < 5; num++) {
                        (it->ppoint)[num] = it->x1 + (it->x2 - it->x1) * landmark_splits[num * 2].at<float>(0);
                        (it->ppoint)[num + 5] = it->y1 + (it->y2 - it->y1) * landmark_splits[num * 2 + 1].at<float>(0);
                    }
                }
                (*bbox).push_back(*it);
            }
        }
        if ((*bbox).size() < 1)
            return;
        if (stage == 1)
        {
            nms((*bbox), nms_thr[stage]);
            refineAndSquareBbox((*bbox), src.rows, src.cols);
        }
        else
        {
            refineAndSquareBbox((*bbox), src.rows, src.cols);
            nms((*bbox), nms_thr[stage], "Min");
        }
    }
}