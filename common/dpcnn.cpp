#include "dpcnn.h"
#include "util.h"
#include "mnn_wrapper.h"

#include <strstream>

#ifdef _DEBUG
#pragma comment(lib,"opencv_world346d.lib")
#else
#pragma comment(lib,"opencv_world346.lib")
#endif
#pragma comment(lib,"MNN.lib")

DPCNN_CPU::DPCNN_CPU()
{
}

DPCNN_CPU::~DPCNN_CPU()
{
    release();
}

bool DPCNN_CPU::newCNN(const string& config_file_path)
{
    std::map<std::string, std::string> configs;
    if (loadConfigFile(config_file_path, configs) != 0)
    {
        std::cerr << "load config file failed" << std::endl;
        return false;
    }

    map<string, string>::iterator model = configs.find("model_file");
    map<string, string>::iterator mean = configs.find("mean_value");
    map<string, string>::iterator scale = configs.find("scale_value");

    if (model == configs.end() ||
        mean == configs.end() ||
        scale == configs.end())
    {
        LOGW("config file corrupted in [%s]\n", config_file_path.c_str());
        return false;
    }

    bool ret = newCNN(configs["model_file"],
        configs["mean_value"],
        configs["scale_value"]);

    return ret;
}

bool DPCNN_CPU::newCNN(const string& model_file,
    const string& mean_value,
    const string& scale_value)
{
    bool ret = true;
    dpcnn = std::static_pointer_cast<void>(std::shared_ptr<MNNWraper>(new MNNWraper));
    if (dpcnn == NULL)
    {
        return false;
    }

    string item;

    std::stringstream scale_vals_ss(scale_value);
    vector<float> scale_vals;
    while (getline(scale_vals_ss, item, ',')) {
        scale_vals.push_back(std::atof(item.c_str()));
    }

    std::stringstream mean_vals_ss(mean_value);
    vector<float> mean_vals;
    while (getline(mean_vals_ss, item, ',')) {
        mean_vals.push_back(std::atof(item.c_str()));
    }

    shared_ptr<MNNWraper> dpcnn_ptr = std::static_pointer_cast<MNNWraper>(dpcnn);
    ret = dpcnn_ptr->init(model_file,
        mean_vals,
        scale_vals);

    return ret;
}

bool DPCNN_CPU::newCNN(const unsigned char model_bin[],
    const int model_size,
    const string& mean_value,
    const string& scale_value)
{
    release();
    bool ret = true;
    dpcnn = std::static_pointer_cast<void>(std::shared_ptr<MNNWraper>(new MNNWraper));
    if (dpcnn == NULL)
    {
        return false;
    }

    string item;

    std::stringstream scale_vals_ss(scale_value);
    vector<float> scale_vals;
    while (getline(scale_vals_ss, item, ',')) {
        scale_vals.push_back(std::atof(item.c_str()));
    }

    std::stringstream mean_vals_ss(mean_value);
    vector<float> mean_vals;
    while (getline(mean_vals_ss, item, ',')) {
        mean_vals.push_back(std::atof(item.c_str()));
    }

    shared_ptr<MNNWraper> dpcnn_ptr = std::static_pointer_cast<MNNWraper>(dpcnn);
    ret = dpcnn_ptr->init_mem(model_bin,
        model_size,
        mean_vals,
        scale_vals);

    return ret;
}

vector<float> DPCNN_CPU::predict(const cv::Mat& img, const int& num_thread, const cv::Size& force_size)
{
    if (dpcnn == NULL)
    {
        return vector<float>();
    }
    if (force_size.area() > 0)
    {
        net_param.dst_sz = force_size;
        net_param.crop_sz = force_size;
    }
    shared_ptr<MNNWraper> dpcnn_ptr = std::static_pointer_cast<MNNWraper>(dpcnn);
    vector<float> ret = dpcnn_ptr->predictFeat(img, num_thread, force_size);
    return ret;
}

vector<cv::Mat> DPCNN_CPU::predictMap(const cv::Mat& img, const int& num_thread, const cv::Size& force_size)
{
    if (dpcnn == NULL)
    {
        return cv::Mat();
    }
    if (force_size.area() > 0)
    {
        net_param.dst_sz = force_size;
        net_param.crop_sz = force_size;
    }
    shared_ptr<MNNWraper> dpcnn_ptr = std::static_pointer_cast<MNNWraper>(dpcnn);
    return dpcnn_ptr->predictMat(img, num_thread, force_size);
}

void DPCNN_CPU::release()
{
    if (dpcnn != NULL)
    {
        shared_ptr<MNNWraper> dpcnn_ptr = std::static_pointer_cast<MNNWraper>(dpcnn);
        dpcnn_ptr.reset();
    }
}

NetworkParam DPCNN_CPU::getNetParam()
{
    shared_ptr<MNNWraper> dpcnn_ptr = std::static_pointer_cast<MNNWraper>(dpcnn);
    net_param.dst_sz = dpcnn_ptr->dst_size;
    net_param.crop_sz = dpcnn_ptr->crop_size;
    return net_param;
}
