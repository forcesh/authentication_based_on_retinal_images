
#ifndef SIMILARITY_H
#define SIMILARITY_H

#pragma once
#include "similarity_abstract.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>


namespace retina
{
	//it uses angular and radial partitions
	//this class is for fast discarding of unequal retinas
	class ARPRetina : public Retina
	{
	public:
		ARPRetina(int angularPartNum_ = 360, int radialPartNum_ = 200);
		~ARPRetina();

		virtual float compareVessels(const cv::Mat& vessels1, const cv::Mat& vessels2);
	
	private:
		int angularPartNum, radialPartNum;

		void dftMagnitude(const std::vector<float>& input, cv::Mat& output);
		void angularPartition(const cv::Mat& vessels, cv::Mat& output, const int numberOfPieces = 72, bool show = false);
		void radialPartition(const cv::Mat& vessels, cv::Mat& output, const int numberOfPieces = 8, bool show = false);
	};

	//it uses phase correlate technique
	class PCRetina : public Retina
	{
	public:
		//iteration_ is number of iterations of phase correlation
		PCRetina(int iteration_ = 3);
		~PCRetina();

		static void getTranslation(const cv::Mat& inputImg1, const cv::Mat& inputImg2, cv::Point2f& shift);
		static void getRot(const cv::Mat& inputImg1, const cv::Mat& inputImg2, float& rotation);
		static void rotAlignment(cv::Mat& inputOutput, const float rotation);
		static void translationAlignment(cv::Mat& inputOutput, const cv::Point2f& shift);

		static float segmentsSimilarity(const cv::Mat& segment1, const cv::Mat& segment2);

		//return cv::Vec4f(rot, x, y, similarity) or
		//return cv::Vec3f(rot, x, y) (if method is private)
		//rot, x, y is for input2 image and they may be inaccurate due to using warpAffine iteratively
		//if you want to get similarity then you should use compareVessels or just take the third element of returned vector
		//omp is used
		cv::Vec4f estimateRigidTransform(const cv::Mat& input1, const cv::Mat& input2);
		//omp is used
		virtual float compareVessels(const cv::Mat& vessels1, const cv::Mat& vessels2);

	private:
		enum ALIGNMENT_METHOD{ ROT_AND_TRANSLATION = 1, TRANSLATION_AND_ROT = 2 };
		int iterationNumber;
		cv::Mat input1_, input2_;

		cv::Vec3f estimateRigidTransform(const cv::Mat& input1, cv::Mat& inpOut2, ALIGNMENT_METHOD method);
		//this method estimates rigid transform iteratively
		//it fills rotXY and sim
		//such interface is made for parallel processing
		//variable order is order of supply input1_ and input2_ to cv::Vec3f estimateRigidTransform
		void PCRetina::estimateRigidTransformIter(std::vector<cv::Vec3f>& rotXY, std::vector<float>& sim,
			int size, ALIGNMENT_METHOD method, bool order);
		
	};

}

#endif