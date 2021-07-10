#include "../face_detect/mtcnn/mtcnn.h"
#include "./fd_example_cls.hpp"
using DPFACE_MTCNN::DPMTCNN;

#ifdef _DEBUG
#pragma comment(lib, "dpfacedetectd.lib")
#pragma comment(lib, "opencv_world346d.lib")
#else
#pragma comment(lib,"dpfacedetect.lib")
#pragma comment(lib, "opencv_world346.lib")
#endif

FaceDetectCls::FaceDetectCls(const std::string& config_file_path)
{
	detector = nullptr;
	detector = std::static_pointer_cast<void>(std::shared_ptr<DPMTCNN>(new DPMTCNN));
	if (detector != nullptr)
	{
		shared_ptr<DPMTCNN> det_ptr = std::static_pointer_cast<DPMTCNN>(detector);
		is_init = det_ptr->init(config_file_path);
	}
}

FaceDetectCls::~FaceDetectCls()
{
	if (is_init)
	{
		shared_ptr<DPMTCNN> det_ptr = std::static_pointer_cast<DPMTCNN>(detector);
		det_ptr.reset();
	}
}

bool FaceDetectCls::detect(const cv::Mat& src, std::vector<cv::Rect2f>& faces, std::vector<float>& scores)
{
	if (is_init)
	{
		shared_ptr<DPMTCNN> det_ptr = std::static_pointer_cast<DPMTCNN>(detector);
		return det_ptr->detect(src, faces, scores);
	}
	else
	{
		return is_init;
	}		
}