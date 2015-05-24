#include "stdafx.h"
#include "similarity.h"



namespace retina
{
	ARPRetina::ARPRetina(int angularPartNum_, int radialPartNum_)
	{
		angularPartNum = angularPartNum_;
		radialPartNum = radialPartNum_;
	}
	ARPRetina::~ARPRetina(){};

	void ARPRetina::dftMagnitude(const std::vector<float>& input, cv::Mat& output)
	{
		int optimHeight = cv::getOptimalDFTSize(input.size());

		std::vector<float> copyInput(input);

		int old_size = copyInput.size();
		for (int i = old_size; i < optimHeight; i++) copyInput.push_back(0.f);

		cv::Mat planes[] = { cv::Mat_<float>(copyInput), cv::Mat::zeros(cv::Size(1, optimHeight), CV_32F) };

		cv::Mat complexI;
		merge(planes, 2, complexI);

		cv::dft(complexI, complexI);

		split(complexI, planes);

		output = cv::Mat::zeros(cv::Size(1, optimHeight), CV_32F);
		magnitude(planes[0], planes[1], output);

	}
	void ARPRetina::angularPartition(const cv::Mat& vessels, cv::Mat& output, const int numberOfPieces, bool show)
	{

		std::vector<float> nonZeroPixsInPiece(numberOfPieces);

		cv::Mat copyVess;
		vessels.copyTo(copyVess);

		cv::Point2i center(cvRound(copyVess.cols / 2.f), cvRound(copyVess.rows / 2.f));

		for (int i = 1; i <= numberOfPieces; i++)
		{
			int length = copyVess.rows > copyVess.cols ? copyVess.rows : copyVess.cols;

			cv::Mat mask = cv::Mat::zeros(copyVess.size(), CV_8UC1);

			cv::Point2i secondPoint, thirdPoint;
			secondPoint.x = (int)cvRound(center.x + length * cos(2 * CV_PI / numberOfPieces * i));
			secondPoint.y = (int)cvRound(center.y + length * sin(2 * CV_PI / numberOfPieces * i));

			thirdPoint.x = (int)cvRound(center.x + length * cos(2 * CV_PI / numberOfPieces * (i - 1)));
			thirdPoint.y = (int)cvRound(center.y + length * sin(2 * CV_PI / numberOfPieces * (i - 1)));

			if (secondPoint.y < 0)
			{
				secondPoint.y = 0;
				length = cvRound(-center.y / sin(2 * CV_PI / numberOfPieces * i));
				secondPoint.x = (int)cvRound(center.x + length * cos(2 * CV_PI / numberOfPieces * i));
			}
			if (secondPoint.y >= vessels.rows)
			{
				secondPoint.y = vessels.rows - 1;
				length = cvRound((secondPoint.y - center.y) / sin(2 * CV_PI / numberOfPieces * i));
				secondPoint.x = (int)cvRound(center.x + length * cos(2 * CV_PI / numberOfPieces * i));
			}
			if (secondPoint.x < 0)
			{
				secondPoint.x = 0;
				length = cvRound(-center.x / cos(2 * CV_PI / numberOfPieces * i));
				secondPoint.y = (int)cvRound(center.y + length * sin(2 * CV_PI / numberOfPieces * i));
			}
			if (secondPoint.x >= vessels.cols)
			{
				secondPoint.x = vessels.cols - 1;
				length = cvRound((secondPoint.x - center.x) / cos(2 * CV_PI / numberOfPieces * i));
				secondPoint.y = (int)cvRound(center.y + length * sin(2 * CV_PI / numberOfPieces * i));
			}

			if (thirdPoint.y < 0)
			{
				thirdPoint.y = 0;
				length = cvRound(-center.y / sin(2 * CV_PI / numberOfPieces * (i - 1)));
				thirdPoint.x = (int)cvRound(center.x + length * cos(2 * CV_PI / numberOfPieces * (i - 1)));
			}
			if (thirdPoint.y >= vessels.rows)
			{
				thirdPoint.y = vessels.rows - 1;
				length = cvRound((thirdPoint.y - center.y) / sin(2 * CV_PI / numberOfPieces * (i - 1)));
				thirdPoint.x = (int)cvRound(center.x + length * cos(2 * CV_PI / numberOfPieces * (i - 1)));
			}
			if (thirdPoint.x < 0)
			{
				thirdPoint.x = 0;
				length = cvRound(-center.x / cos(2 * CV_PI / numberOfPieces * (i - 1)));
				thirdPoint.y = (int)cvRound(center.y + length * sin(2 * CV_PI / numberOfPieces * (i - 1)));
			}
			if (thirdPoint.x >= vessels.cols)
			{
				thirdPoint.x = vessels.cols - 1;
				length = cvRound((thirdPoint.x - center.x) / cos(2 * CV_PI / numberOfPieces * (i - 1)));
				thirdPoint.y = (int)cvRound(center.y + length * sin(2 * CV_PI / numberOfPieces * (i - 1)));
			}

			cv::Point triangle[4];
			triangle[0] = center;
			triangle[1] = secondPoint;
			triangle[2] = thirdPoint;
			int size = 3;

			if (((secondPoint.y & thirdPoint.y) == 0) && ((secondPoint.y | thirdPoint.y) != 0))
			{
				size++;
				triangle[2].y = 0;
				if (secondPoint.y == 0) triangle[2].x = thirdPoint.x;
				else triangle[2].x = secondPoint.x;
				triangle[3] = thirdPoint;
			}

			if ((secondPoint.y != vessels.rows - 1) || (thirdPoint.y != vessels.rows - 1))
			{
				if (((secondPoint.y == vessels.rows - 1) && (thirdPoint.y != vessels.rows - 1)) ||
					((thirdPoint.y == vessels.rows - 1) && (secondPoint.y != vessels.rows - 1)))
				{
					size++;
					triangle[2].y = vessels.rows - 1;
					if (secondPoint.y == vessels.rows - 1) triangle[2].x = thirdPoint.x;
					else triangle[2].x = secondPoint.x;
					triangle[3] = thirdPoint;
				}
			}

			const cv::Point* ppt[1] = { triangle };
			int npt[] = { size };

			fillPoly(mask, ppt, npt, 1, cv::Scalar(255));

			bitwise_and(mask, vessels, mask);
			nonZeroPixsInPiece[i - 1] = (float)countNonZero(mask);

			if (show)
			{
				line(copyVess, center, secondPoint, cv::Scalar(255), 1);
				line(copyVess, center, thirdPoint, cv::Scalar(255), 1);
			}
		}

		dftMagnitude(nonZeroPixsInPiece, output);

		if (show)
		{
			imshow("vessels", copyVess);
			cv::waitKey();
		}

	}
	void ARPRetina::radialPartition(const cv::Mat& vessels, cv::Mat& output, const int numberOfPieces, bool show)
	{
		std::vector<float> nonZeroPixsInPiece(numberOfPieces);

		cv::Mat copyVess;
		vessels.copyTo(copyVess);

		cv::Point2i center(cvRound(copyVess.cols / 2.f), cvRound(copyVess.rows / 2.f));

		float max_radius = (float)(vessels.rows < vessels.cols ? vessels.rows / 2 : vessels.cols / 2);

		for (int i = 0; i < numberOfPieces; i++)
		{

			int currRadius = cvRound(max_radius / numberOfPieces * (i + 1));
			int prevRadius = cvRound(max_radius / numberOfPieces * (i));

			if (show)
			{
				circle(copyVess, center, currRadius, cv::Scalar(255));
			}

			cv::Mat mask = cv::Mat::zeros(copyVess.size(), CV_8UC1);

			if (i > 0)
			{
				circle(mask, center, currRadius, cv::Scalar(255), CV_FILLED);
				circle(mask, center, prevRadius, cv::Scalar(0), CV_FILLED);
			}
			else
			{
				circle(mask, center, currRadius, cv::Scalar(255), CV_FILLED);
			}

			bitwise_and(mask, vessels, mask);
			nonZeroPixsInPiece[i] = (float)countNonZero(mask);

		}

		dftMagnitude(nonZeroPixsInPiece, output);

		if (show)
		{
			imshow("vessels", copyVess);
			cv::waitKey();
		}

	}
	float ARPRetina::compareVessels(const cv::Mat& vessels1, const cv::Mat& vessels2)
	{
		if ((vessels1.channels() > 1) || (vessels2.channels() > 1) ||
			(vessels1.size() != vessels2.size()))
		{
			std::cerr << "(vessels1.channels() > 1) || (vessels2.channels() > 1) ||"
				"(vessels1.size() != vessels2.size())" << std::endl;

			return -1.f;
		}

		cv::Mat angPart1, angPart2;
		angularPartition(vessels1, angPart1, angularPartNum);
		angularPartition(vessels2, angPart2, angularPartNum);

		cv::Mat radPart1, radPart2;

		radialPartition(vessels1, radPart1, radialPartNum);
		radialPartition(vessels2, radPart2, radialPartNum);

		cv::Mat radDiff, angDiff;

		cv::absdiff(radPart1, radPart2, radDiff);
		cv::absdiff(angPart1, angPart2, angDiff);

		float sim = 0;

		angDiff /= 1000.f;
		radDiff /= 1000.f;

		for (int i = 0; i < angDiff.rows; i++) sim += angDiff.at<float>(i, 0);
		for (int i = 0; i < radDiff.rows; i++) sim += radDiff.at<float>(i, 0);

		sim /= (angDiff.rows + radDiff.rows);
		sim = 1 - sim;

		return sim;
	}

	PCRetina::PCRetina(int iterationNumber_)
	{
		iterationNumber = iterationNumber_;
	}
	PCRetina::~PCRetina(){};

	void PCRetina::translationAlignment(cv::Mat& inputOutput, const cv::Point2f& shift)
	{

		cv::Mat shiftMap(2, 3, CV_32FC1);
		cv::Point2f srcTri[3];
		cv::Point2f dstTri[3];
		float x = (float)shift.x;
		float y = (float)shift.y;

		srcTri[0] = cv::Point2f(0, 0);
		srcTri[1] = cv::Point2f((float)inputOutput.cols - 1, 0);
		srcTri[2] = cv::Point2f(0, (float)inputOutput.rows - 1);

		dstTri[0] = cv::Point2f(x, y);
		dstTri[1] = cv::Point2f((float)inputOutput.cols - 1 + x, y);
		dstTri[2] = cv::Point2f(x, (float)inputOutput.rows - 1 + y);

		shiftMap = getAffineTransform(srcTri, dstTri);
		warpAffine(inputOutput, inputOutput, shiftMap, inputOutput.size());
	}
	void PCRetina::getTranslation(const cv::Mat& inputImg1, const cv::Mat& inputImg2, cv::Point2f& shift)
	{
		if (inputImg1.size() != inputImg2.size())
		{
			std::cerr << "inputImg1.size() != inputImg2.size()" << std::endl;
			return;
		}

		cv::Mat hann, img1_64, img2_64;

		int M = cv::getOptimalDFTSize(inputImg1.rows);
		int N = cv::getOptimalDFTSize(inputImg1.cols);

		cv::Mat padded1, padded2;

		if (inputImg1.channels() > 1)
			cvtColor(inputImg1, padded1, CV_BGR2GRAY);
		else
			padded1 = inputImg1;

		if (inputImg2.channels() > 1)
			cvtColor(inputImg2, padded2, CV_BGR2GRAY);
		else
			padded2 = inputImg2;

		cv::copyMakeBorder(padded1, padded1, 0, M - inputImg1.rows, 0, N - inputImg1.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		cv::copyMakeBorder(padded2, padded2, 0, M - inputImg2.rows, 0, N - inputImg2.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		cv::createHanningWindow(hann, padded2.size(), CV_64F);

		padded1.convertTo(img1_64, CV_64F);
		padded2.convertTo(img2_64, CV_64F);

		shift = cv::phaseCorrelate(img1_64, img2_64, hann);
		shift.x = -cvRound(shift.x);
		shift.y = -cvRound(shift.y);

	}
	void PCRetina::getRot(const cv::Mat& inputImg1, const cv::Mat& inputImg2, float& rotation)
	{

		if (inputImg1.size() != inputImg2.size())
		{
			std::cerr << "inputImg1.size() != inputImg2.size()" << std::endl;
			return;
		}

		cv::Mat pImg1 = cv::Mat::zeros(inputImg1.size(), CV_8UC1);
		cv::Mat pImg2 = cv::Mat::zeros(inputImg2.size(), CV_8UC1);

		cv::Mat gray1, gray2;

		if (inputImg1.channels() > 1)
			cvtColor(inputImg1, gray1, CV_BGR2GRAY);
		else
			gray1 = inputImg1;

		if (inputImg2.channels() > 1)
			cvtColor(inputImg2, gray2, CV_BGR2GRAY);
		else
			gray2 = inputImg2;

		IplImage ipImg1 = gray1, ipl_p1 = pImg1;
		IplImage ipImg2 = gray2, ipl_p2 = pImg2;

		float mLog = 40;

		cvLogPolar(&ipImg1, &ipl_p1, cv::Point2f(inputImg1.cols / 2.f, inputImg1.rows / 2.f), mLog);
		cvLogPolar(&ipImg2, &ipl_p2, cv::Point2f(inputImg2.cols / 2.f, inputImg2.rows / 2.f), mLog);

		cv::Mat logPolImg1(&ipl_p1);
		cv::Mat logPolImg2(&ipl_p2);

		cv::Mat hann, lpImg164f, lpImg264f;

		int M = cv::getOptimalDFTSize(logPolImg1.rows);
		int N = cv::getOptimalDFTSize(logPolImg1.cols);

		cv::Mat padded1, padded2;
		cv::copyMakeBorder(logPolImg1, padded1, 0, M - logPolImg1.rows, 0, N - logPolImg1.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		cv::copyMakeBorder(logPolImg2, padded2, 0, M - logPolImg1.rows, 0, N - logPolImg1.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		createHanningWindow(hann, padded2.size(), CV_64F);

		padded1.convertTo(lpImg164f, CV_64F);
		padded2.convertTo(lpImg264f, CV_64F);

		cv::Point2f shift = phaseCorrelate(lpImg164f, lpImg264f, hann);

		/*scale = 1 / (expf(shift.x / mLog));
		scale = cvRound(scale * 10.f);
		scale /= 10;*/

		//systematic error
		//if (shift.y > 0) shift.y += 0.2;
		//else shift.y -= 0.2;

		rotation = cvRound(shift.y / lpImg264f.rows * 360.f);
	}
	void PCRetina::rotAlignment(cv::Mat& inputOutput, const float rotation){

		cv::Point2f center(inputOutput.cols / 2.f, inputOutput.rows / 2.f);
		cv::Mat rotMap(2, 3, CV_32FC1);

		rotMap = cv::getRotationMatrix2D(center, rotation, 1);
		cv::warpAffine(inputOutput, inputOutput, rotMap, inputOutput.size());
	}

	float PCRetina::segmentsSimilarity(const cv::Mat& segment1, const cv::Mat& segment2)
	{

		if ((segment1.channels() > 1) || (segment2.channels() > 1) ||
			(segment1.size() != segment2.size()))
		{
			std::cerr << "(segment1.channels() > 1) || (segment2.channels() > 1) ||"
				"(segment1.size() != segment2.size())" << std::endl;

			return -1.f;
		}

		cv::Mat intersection, union_;
		bitwise_and(segment1, segment2, intersection);
		bitwise_or(segment1, segment2, union_);

		int hits = countNonZero(intersection);
		int hits_max = countNonZero(union_);

		return (float)hits / hits_max;
	}

	cv::Vec3f PCRetina::estimateRigidTransform(const cv::Mat& input1, cv::Mat& inpOut2, ALIGNMENT_METHOD method)
	{

		float rotation;
		cv::Point2f shift;

		if (method == TRANSLATION_AND_ROT)
		{
			getTranslation(input1, inpOut2, shift);
			translationAlignment(inpOut2, shift);
			threshold(inpOut2, inpOut2, 150, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			getRot(input1, inpOut2, rotation);
			rotAlignment(inpOut2, rotation);
			threshold(inpOut2, inpOut2, 150, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		}
		else if (method == ROT_AND_TRANSLATION)
		{
			getRot(input1, inpOut2, rotation);
			rotAlignment(inpOut2, rotation);
			threshold(inpOut2, inpOut2, 150, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			getTranslation(input1, inpOut2, shift);
			translationAlignment(inpOut2, shift);
			threshold(inpOut2, inpOut2, 150, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		}

		cv::Vec3f rotXY(rotation, shift.x, shift.y);

		return rotXY;
	}
	void PCRetina::estimateRigidTransformIter(std::vector<cv::Vec3f>& rotXY, std::vector<float>& sim,
		int size, ALIGNMENT_METHOD method, bool order)
	{

		cv::Mat copy1, copy2;

		if (order)
		{
			input2_.copyTo(copy2);
		}
		else
		{
			input1_.copyTo(copy1);
		}

		for (int i = 0; i < size; i++)
		{
			if (order)
			{
				rotXY.push_back(estimateRigidTransform(input1_, copy2, method));
				sim.push_back(segmentsSimilarity(input1_, copy2));
			}
			else
			{
				rotXY.push_back(estimateRigidTransform(input2_, copy1, method));
				sim.push_back(segmentsSimilarity(input2_, copy1));
			}
		}
	}
	cv::Vec4f PCRetina::estimateRigidTransform(const cv::Mat& input1, const cv::Mat& input2)
	{

		if ((input1.channels() > 1) || (input2.channels() > 1) ||
			(input1.size() != input2.size()))
		{
			std::cerr << "(input1.channels() > 1) || (input2.channels() > 1) ||"
				"(input1.size() != input2.size())" << std::endl;

			return cv::Vec4f(-1, -1, -1, -1);
		}

		input1_ = input1;
		input2_ = input2;

		std::vector<float> sim[4];
		std::vector<cv::Vec3f> rotXY[4];

		#pragma omp parallel for
		for (int i = 0; i < 4; i++)
		{
			int size;
			ALIGNMENT_METHOD method;
			bool order;

			if (i > 1) order = false;
			else order = true;

			if (i % 2 == 0)
			{
				size = iterationNumber - 1;
				method = TRANSLATION_AND_ROT;
			}
			else
			{
				size = iterationNumber;
				method = ROT_AND_TRANSLATION;
			}

			estimateRigidTransformIter(rotXY[i], sim[i], size, method, order);

		}

		float maxSim = sim[0][0];
		int idx = 0, idy = 0;

		for (int j = 0; j < 4; j++)
		{
			for (unsigned int i = 0; i < sim[j].size(); i++)
			{
				if ((i + j) == 0) continue;

				if (sim[j][i] > maxSim)
				{
					maxSim = sim[j][i];
					idx = i;
					idy = j;
				}
			}
		}

		cv::Vec4f rotXYSim(0, 0, 0, 0);
		rotXYSim[3] = maxSim;

		if (idy == 0 || idy == 1)
		{
			while (idx >= 0)
			{
				rotXYSim[0] += rotXY[idy][idx][0];
				rotXYSim[1] += rotXY[idy][idx][1];
				rotXYSim[2] += rotXY[idy][idx][2];
				idx--;
			}
		}
		else
		{
			while (idx >= 0)
			{
				rotXYSim[0] -= rotXY[idy][idx][0];
				rotXYSim[1] -= rotXY[idy][idx][1];
				rotXYSim[2] -= rotXY[idy][idx][2];
				idx--;
			}
		}

		return rotXYSim;
	}
	float PCRetina::compareVessels(const cv::Mat& vessels1, const cv::Mat& vessels2)
	{
		cv::Vec4f rotXYSim = estimateRigidTransform(vessels1, vessels2);

		return rotXYSim[3];
	}
}
