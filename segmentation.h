
#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#pragma once
#include <iostream>
#include <queue>

#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



namespace retina
{
	enum PERFORMANCE_PARAMS{ TPR = 1, FPR = 2, SPC = 4, ACC = 8, F_SCORE = 16 };

	//it's for binary images
	//parameters : TpTnFpFn
	void countTpTnFpFn(const cv::Mat& gold_standard, const cv::Mat& extracted_segment, std::vector<int>& tpTnFpFn);
	//it's for binary images
	//return PERFORMANCE_PARAMS : TPR, FPR, SPC, ACC, F_SCORE
	std::vector<float> countPerformanceParameters(const cv::Mat& gold_standard, const cv::Mat& extracted_segment,
		int performance_flags, bool showTpTnFpFnPN = false);
	//return PERFORMANCE_PARAMS : TPR, FPR, SPC, ACC, F_SCORE
	//parameters : TpTnFpFn
	std::vector<float> TpTnFpFn2PerformanceParameters(std::vector<int>& tpTnFpFn, int performance_flags, bool showTpTnFpFnPN = false);

	//it's for binary images
	void filterByLength(const cv::Mat & input, cv::Mat & output, unsigned int length = 10);

	//if specificity_param == 1 then specificity will be high
	void vessSegmentation(const cv::Mat& input, cv::Mat& output, bool specificity_param = true, bool show = false);
	//sensitivity_param is between 0 and 1
	//if parameter is close to 1 then TPR will be high
	//else specificity will be high
	//if specificity_param == 1 then specificity will be high
	//omp is used
	void vessSegmentationGabor(const cv::Mat& input, cv::Mat& output, bool specificity_param = true, float sensitivity_param = 0.34f,
		bool show = false);
	
	//it's for binary images
	void vesselsThiningMORPH(const cv::Mat& input, cv::Mat& output);
	
	//
	cv::Point2i opticDiscLocalization(const cv::Mat& input, cv::Mat& output, bool show = false);

	//it's only for retinas like in the RIM database
	//this method will show good results if color space is BGR or if it is lightness channel
	void opticDiscSegmentationRIM(const cv::Mat& input, cv::Mat& output, bool show = false);
	//it uses opticDiscSegmentationRIM and opticDiscLocalization
	//for more details see them
	void opticDiscSegmentation(const cv::Mat& input, cv::Mat& output, bool show = false);
}

#endif