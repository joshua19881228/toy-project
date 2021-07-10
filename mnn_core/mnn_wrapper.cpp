#include "mnn_wrapper.h"
#include "util.h"
#include <opencv2/imgproc/types_c.h>

inline cv::Mat cvtNcnnMat2CvMat32F(const MNN::Tensor& src)
{
    if (halide_type_float != src.getType().code)
    {
        LOGE("input is not float type");
        return cv::Mat();
    }
    int ch = src.channel();
    int h = std::max(1, src.height());
    int w = std::max(1, src.width());
    std::vector<cv::Mat> split(src.channel());
    for (int c = 0; c < ch; ++c)
    {
        float* ptr = src.host<float>() + c * h * w;
        split[c] = cv::Mat(h, w, CV_32FC1, ptr);
    }
    cv::Mat dst;
    cv::merge(split, dst);
    return dst.clone();
}

MNNWraper::MNNWraper()
{
    means.clear();
    scales.clear();
}

MNNWraper::~MNNWraper()
{
    release(true);
}

void MNNWraper::release(bool rls_net)
{
    if (rls_net)
        net.reset();
    means.clear();
    scales.clear();
}

bool MNNWraper::checkParams()
{
    bool ret = true;
    if (means.size() != scales.size())
    {
        LOGE("means.size() == scales.size() [%d != %d] ", means.size(), scales.size());
        ret = ret && false;
    }

    if (dst_size.height < crop_size.height)
    {
        LOGE("dst_size.height >= crop_size.height [%d < %d] ", dst_size.height, crop_size.height);
        ret = ret && false;
    }

    if (dst_size.width < crop_size.width)
    {
        LOGE("dst_size.width >= crop_size.width [%d < %d] ", dst_size.width, crop_size.width);
        ret = ret && false;
    }
    return ret;
}

bool MNNWraper::init(const std::string& _param_path,
    const std::vector<float>& _means,
    const std::vector<float>& _scales)
{
    release(true);
    bool ret = true;

    net.reset(MNN::Interpreter::createFromFile(_param_path.c_str()));
    //std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(_param_path.c_str()));
    if (net == nullptr)
        ret = false;
    means = _means;
    scales = _scales;

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    session = net->createSession(config);
    auto input = net->getSessionInput(session, NULL);
    dst_size = cv::Size(input->width(), input->height());
    crop_size = dst_size;

    ret = ret && checkParams();
    return ret;
}

bool MNNWraper::init_mem(const unsigned char* model_bin,
    const int model_size,
    const std::vector<float>& _means,
    const std::vector<float>& _scales)
{
    release(true);
    bool ret = true;
    net.reset(MNN::Interpreter::createFromBuffer(model_bin, model_size));
    if (net == nullptr)
        return false;
    means = _means;
    scales = _scales;

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_AUTO;
    session = net->createSession(config);
    auto input = net->getSessionInput(session, NULL);
    dst_size = cv::Size(input->width(), input->height());
    crop_size = dst_size;

    ret = ret && checkParams();
    return ret;
}

std::vector<cv::Mat> MNNWraper::predictMat(const cv::Mat& src, const int& num_thread, const cv::Size& force_size)
{
    std::vector<cv::Mat> ret_cvmat;
    cv::Mat input_cvmat;
    auto input = net->getSessionInput(session, NULL);
    dst_size = cv::Size(input->width(), input->height());
    //resize
    if (force_size.area() != 0)
    {
        dst_size = force_size;
        crop_size = force_size;
    }
    cv::resize(src, input_cvmat, dst_size);
#ifdef _DEBUG
    LOGI("input size is %d x %d", input->width(), input->height());
#endif // DEBUG
    auto outputs = net->getSessionOutputAll(session);

    std::vector<int> inputDims = { 1, 3, input_cvmat.rows, input_cvmat.cols };
    net->resizeTensor(input, inputDims);
    net->resizeSession(session);
    //convert color
    if (means.size() == 1 && input_cvmat.channels() == 3)
    {
        cv::cvtColor(input_cvmat, input_cvmat, CV_BGR2GRAY);
    }
    if (means.size() != input_cvmat.channels())
    {
        LOGE("means.size() == src.channels() [%d != %d]", means.size(), input_cvmat.channels());
        return std::vector<cv::Mat>();
    }

    //subtract mean and scale
    if (input_cvmat.channels() == 3)
    {
        MNN::CV::Matrix trans;
        trans.postScale(1.0 / input->width(), 1.0 / input->height());
        trans.postScale(input_cvmat.cols, input_cvmat.rows);

        MNN::CV::ImageProcess::Config config_ip;
        config_ip.filterType = MNN::CV::BILINEAR;
        float mean[3] = { means[0], means[1], means[2] };
        float normals[3] = { scales[0], scales[1], scales[2] };
        ::memcpy(config_ip.mean, mean, sizeof(mean));
        ::memcpy(config_ip.normal, normals, sizeof(normals));
        config_ip.sourceFormat = MNN::CV::BGR;
        config_ip.destFormat = MNN::CV::BGR;

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config_ip));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)input_cvmat.data, input_cvmat.cols, input_cvmat.rows, 0, input);
    }
    else if (input_cvmat.channels() == 1)
    {
        MNN::CV::Matrix trans;
        trans.postScale(1.0 / input->width(), 1.0 / input->height());
        trans.postScale(input_cvmat.cols, input_cvmat.rows);

        MNN::CV::ImageProcess::Config config_ip;
        config_ip.filterType = MNN::CV::BILINEAR;
        float mean[1] = { means[0] };
        float normals[1] = { scales[0] };
        ::memcpy(config_ip.mean, mean, sizeof(mean));
        ::memcpy(config_ip.normal, normals, sizeof(normals));
        config_ip.sourceFormat = MNN::CV::GRAY;
        config_ip.destFormat = MNN::CV::GRAY;

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config_ip));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)input_cvmat.data, input_cvmat.cols, input_cvmat.rows, 0, input);
    }
    else
    {
        LOGE("not defined number of channels");
        return std::vector<cv::Mat>();
    }

    net->runSession(session);

    for (auto output : outputs)
    {
#ifdef _DEBUG
        LOGI(output.first.c_str());
        LOGI("   size %d x %d x %d", output.second->channel(), output.second->width(), output.second->height());
#endif // DEBUG
        auto dimType = output.second->getDimensionType();
        if (output.second->getType().code != halide_type_float) {
            dimType = MNN::Tensor::TENSORFLOW;
        }
        std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output.second, dimType));//outputuser??
        //MNN_PRINT("output size:%d\n", outputUser->elementSize());
        output.second->copyToHostTensor(outputUser.get());
        auto type = outputUser->getType();
        auto size = outputUser->elementSize();
        ret_cvmat.push_back(cvtNcnnMat2CvMat32F(*outputUser));
    }

    return ret_cvmat;
}

std::vector<float> MNNWraper::predictFeat(const cv::Mat& src, const int& num_thread, const cv::Size& force_size)
{
    std::vector<float> ret;
    cv::Mat input_cvmat;
    auto input = net->getSessionInput(session, NULL);
    dst_size = cv::Size(input->width(), input->height());
    //resize
    if (force_size.area() != 0)
    {
        dst_size = force_size;
        crop_size = force_size;
    }
    cv::resize(src, input_cvmat, dst_size);

#ifdef _DEBUG
    LOGI("input size is %d x %d", input->width(), input->height());
#endif // DEBUG

    auto outputs = net->getSessionOutputAll(session);

    std::vector<int> inputDims = { 1, 3, input_cvmat.rows, input_cvmat.cols };
    net->resizeTensor(input, inputDims);
    net->resizeSession(session);

    //convert color
    if (means.size() == 1 && input_cvmat.channels() == 3)
    {
        cv::cvtColor(input_cvmat, input_cvmat, CV_BGR2GRAY);
    }
    if (means.size() != input_cvmat.channels())
    {
        LOGE("means.size() == src.channels() [%d != %d]", means.size(), input_cvmat.channels());
        return std::vector<float>();
    }

    if (input_cvmat.channels() == 3)
    {
        MNN::CV::Matrix trans;
        trans.postScale(1.0 / input->width(), 1.0 / input->height());
        trans.postScale(input_cvmat.cols, input_cvmat.rows);

        MNN::CV::ImageProcess::Config config_ip;
        config_ip.filterType = MNN::CV::BILINEAR;
        float mean[3] = { means[0], means[1], means[2] };
        float normals[3] = { scales[0], scales[1], scales[2] };
        ::memcpy(config_ip.mean, mean, sizeof(mean));
        ::memcpy(config_ip.normal, normals, sizeof(normals));
        config_ip.sourceFormat = MNN::CV::BGR;
        config_ip.destFormat = MNN::CV::BGR;

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config_ip));
        pretreat->setMatrix(trans);
        pretreat->convert(input_cvmat.data, input_cvmat.cols, input_cvmat.rows, 0, input);
    }
    else if (input_cvmat.channels() == 1)
    {
        MNN::CV::Matrix trans;
        trans.postScale(1.0 / input->width(), 1.0 / input->height());
        trans.postScale(input_cvmat.cols, input_cvmat.rows);

        MNN::CV::ImageProcess::Config config_ip;
        config_ip.filterType = MNN::CV::BILINEAR;
        float mean[1] = { means[0] };
        float normals[1] = { scales[0] };
        ::memcpy(config_ip.mean, mean, sizeof(mean));
        ::memcpy(config_ip.normal, normals, sizeof(normals));
        config_ip.sourceFormat = MNN::CV::GRAY;
        config_ip.destFormat = MNN::CV::GRAY;

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config_ip));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)input_cvmat.data, input_cvmat.cols, input_cvmat.rows, 0, input);
    }
    else
    {
        LOGE("not defined number of channels");
        return std::vector<float>();
    }

    net->runSession(session);

    for (auto output : outputs)
    {
#ifdef _DEBUG
        LOGI(output.first.c_str());
        LOGI("   size %d x %d x %d", output.second->channel(), output.second->width(), output.second->height());
#endif // DEBUG

        auto dimType = output.second->getDimensionType();
        if (output.second->getType().code != halide_type_float) {
            dimType = MNN::Tensor::TENSORFLOW;
        }
        std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output.second, dimType));//outputuser??
        //MNN_PRINT("output size:%d\n", outputUser->elementSize());
        output.second->copyToHostTensor(outputUser.get());
        auto type = outputUser->getType();
        auto size = outputUser->elementSize();
        auto values = outputUser->host<float>();
        for (int j = 0; j < size; j++)
        {
            ret.push_back(values[j]);
        }
    }

    return ret;
}