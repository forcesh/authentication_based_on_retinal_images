
#pragma once
#include <opencv2/core/core.hpp>


namespace retina
{
	class Retina
	{
	public:
		virtual ~Retina(){};
		//for CV_8UC1
		virtual float compareVessels(const cv::Mat& vessels1, const cv::Mat& vessels2) = 0;
	};
}