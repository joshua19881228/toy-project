#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

class DLLEXPORT FaceDetectCls
{
public:
	FaceDetectCls(const std::string& config_file_path);
	~FaceDetectCls();
	bool detect(const cv::Mat& src, std::vector<cv::Rect2f>& faces, std::vector<float>& scores);
private:
	std::shared_ptr<void> detector;
	bool is_init;
};