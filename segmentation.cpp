#include "stdafx.h"
#include "segmentation.h"


//bgr; hls, hsv etc.
enum COLORS{first = 0, second = 1, third = 2};
void splitColorImg(const cv::Mat& input, cv::Mat& output, COLORS color = second)
{

	std::vector<cv::Mat> xyz_planes;

	split(input, xyz_planes);

	xyz_planes[color].copyTo(output);
}

cv::Mat calcHist(const cv::Mat& input, const cv::Mat& mask)
{
	cv::Mat hist;

	int histSize = 256;

	/// Set the ranges
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	/// Compute the histograms:

	calcHist(&input, 1, 0, mask, hist, 1, &histSize, &histRange, uniform, accumulate);

	normalize(hist, hist, 0, 100, cv::NORM_MINMAX, -1, cv::Mat());

	return hist;
}

void backgroundExclusion(const cv::Mat& input, cv::Mat& output, cv::Mat& outputMask, double clipLimit1 = 15,
	cv::Size size4clahe1 = cv::Size(15, 15), double clipLimit2 = 15, cv::Size size4clahe2 = cv::Size(15, 15))
{

	cv::Mat copyInput;
	input.copyTo(copyInput);

	if (copyInput.channels() > 1)
		splitColorImg(copyInput, copyInput, second);

	threshold(copyInput, outputMask, 40, 255, CV_THRESH_BINARY_INV);

	cv::Ptr < cv::CLAHE > clahe = cv::createCLAHE(clipLimit1, size4clahe1);
	clahe.obj->apply(copyInput, copyInput);
	
	int morph_size = 3;
	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
		cv::Point(morph_size, morph_size));
	morphologyEx(outputMask, outputMask, cv::MORPH_OPEN, element);

	blur(copyInput, output, cv::Size(14, 14));
	output -= copyInput;

	output -= outputMask;

	clahe = cv::createCLAHE(clipLimit2, size4clahe2);
	clahe.obj->apply(output, output);
}

float Gausian2D(int x, int y, float sigma, float gamma, float theta)
{
	float xr = x*cos(theta) + y*sin(theta);
	float yr = -x*sin(theta) + y*cos(theta);

	yr *= gamma;

	float res;

	float sigma22 = 2 * sigma*sigma;
	float A = 1 / sqrt(CV_PI*sigma22);
	float A2 = A*A;

	res = A2*exp(-((xr*xr) + (yr*yr)) / sigma22);

	return res;
}
cv::Mat GaborKernel(int Sx, int Sy, float lambda, float theta, float sigma, float psi = 0, float gamma = 1)
{
	float *Greal;
	float *Gimag;
	int szx, szy;

	float b = 0.0f;
	szx = Sx * 2 + 1;
	szy = Sy * 2 + 1;
	float omega0 = 1.0f / lambda;


	float omegax = omega0*cosf(theta) + psi;
	float omegay = omega0*sinf(theta) + psi;

	Greal = (float*)malloc(szx*szy*sizeof(float));
	Gimag = (float*)malloc(szx*szy*sizeof(float));

	for (int x = -Sx; x <= Sx; x++)
	{
		for (int y = -Sy; y <= Sy; y++)
		{
			b = 2 * CV_PI*(omegax*x + omegay*y);
			Greal[(Sy + y)*szx + Sx + x] = Gausian2D(x, y, sigma, gamma, theta)*cosf(b);
			Gimag[(Sy + y)*szx + Sx + x] = Gausian2D(x, y, sigma, gamma, theta)*sinf(b);
		}
	}
	cv::Mat seReIm[2];
	seReIm[0] = cv::Mat(szy, szx, CV_32FC1, Greal);
	seReIm[1] = cv::Mat(szy, szx, CV_32FC1, Gimag);
	cv::Mat se = cv::Mat(szy, szx, CV_32FC2);
	merge(seReIm, 2, se);
	return se;
}
void GaborFilter4vessels(const cv::Mat& input, cv::Mat& output, int thetaStep = 10, bool showKernel = false)
{
	int Sx = 15, Sy = 15;
	float lambda = 4.f;
	float sigma = 1.f, psi = 0, gamma = 0.15f;

	std::vector<cv::Mat> filteredImages(int(180.f / thetaStep) + 1);

	#pragma omp parallel for
	for (int theta = 0; theta < 180; theta += thetaStep)
	{
		cv::Mat kernel = GaborKernel(Sx, Sy, lambda, theta * CV_PI / 180, sigma, psi, gamma);

		cv::Mat kernelImRe[2];
		split(kernel, kernelImRe);

		cv::Mat resultRe = cv::Mat(input.rows, input.cols, CV_32FC1);
		cv::Mat resultIm = cv::Mat(input.rows, input.cols, CV_32FC1);
		cv::Mat resultMag = cv::Mat(input.rows, input.cols, CV_32FC1);

		filter2D(input, resultRe, CV_32FC1, kernelImRe[0]);

		filter2D(input, resultIm, CV_32FC1, kernelImRe[1]);

		magnitude(resultRe, resultIm, resultMag);

		if (showKernel)
		{
			double M = 0, m = 0;
			minMaxLoc(kernelImRe[0], &m, &M);
			if ((M - m) > 0)	{ kernelImRe[0] = kernelImRe[0] * (1.0 / (M - m)) - m / (M - m); }
			cv::imshow("KernelRe", kernelImRe[0]);

			minMaxLoc(kernelImRe[1], &m, &M);
			if ((M - m) > 0)	{ kernelImRe[1] = kernelImRe[1] * (1.0 / (M - m)) - m / (M - m); }
			cv::imshow("KernelIm", kernelImRe[1]);

			cv::waitKey();
		}

		int index = theta / thetaStep;
		filteredImages[index] = resultMag;

	}

	for (unsigned int idx = 1; idx < filteredImages.size(); idx++)
	{
		for (int i = 0; i < filteredImages[idx].rows; i++)
		{
			for (int j = 0; j < filteredImages[idx].cols; j++)
			{
				if (filteredImages[idx].at<float>(i, j) > filteredImages[0].at<float>(i, j))//max
				{
					filteredImages[0].at<float>(i, j) = filteredImages[idx].at<float>(i, j);
				}
			}
		}
	}

	double M = 0, m = 0;
	minMaxLoc(filteredImages[0], &m, &M);
	if ((M - m) > 0)	{ filteredImages[0] = filteredImages[0] * (1.0 / (M - m)) - m / (M - m); }

	filteredImages[0].convertTo(filteredImages[0], CV_8UC1, 255, 0);

	filteredImages[0] = 255 - filteredImages[0];

	filteredImages[0].copyTo(output);
}

int getThresholdUsingHist(const cv::Mat& input, const cv::Mat& mask, float sensitivity_param = 0.34f, bool show = false)
{

	cv::Mat hist = calcHist(input, mask);

	int max = 0;

	for (int i = 1; i < hist.rows; i++)
	{
		if (hist.at<float>(i) > hist.at<float>(max))
			max = i;

	}

	float limit = hist.at<float>(max) * sensitivity_param;

	int thresholdValue = 0;

	for (int i = max - 1; i >= 0; i--)
	{
		if (hist.at<float>(i) <= limit)
		{
			thresholdValue = i;
			break;
		}
	}

	if (show)
	{

		int hist_w = 512; int hist_h = 400;

		normalize(hist, hist, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());

		int bin_w = cvRound((double)hist_w / hist.rows);

		cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

		int thresholdValueInImg = cvRound(thresholdValue / (float)hist.rows * hist_w);

		line(histImage, cv::Point(thresholdValueInImg, 0), cv::Point(thresholdValueInImg, hist_h - 1), cv::Scalar(255, 0, 0));


		for (int i = 1; i < hist.rows; i++)
		{
			line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
				cv::Scalar(255, 255, 255), 2, 8, 0);
		}

		cv::imshow("hist", histImage);
		cv::waitKey();
	}

	return thresholdValue;
}

std::vector<float> params2performanceParams(std::vector<int>& parameters, int performance_flags)
{
	const int numberOfFlags = 5;//number of PERFORMANCE_PARAMS

	std::vector<float> result(numberOfFlags);

	for (int i = 0; i < numberOfFlags; i++)
	{
		result[i] = -1;
	}

	if ((performance_flags & retina::TPR) == retina::TPR)
		result[0] = (float)parameters[0] / (parameters[0] + parameters[3]);// TP / ( TP + FN)

	if ((performance_flags & retina::FPR) == retina::FPR)
		result[1] = (float)parameters[2] / (parameters[2] + parameters[1]);// FP / ( FP + TN)

	if ((performance_flags & retina::SPC) == retina::SPC)
		result[2] = 1 - result[1];// TN / ( FP + TN )

	if ((performance_flags & retina::ACC) == retina::ACC)
		result[3] = (float)(parameters[0] + parameters[1]) /
		(parameters[0] + parameters[3] + parameters[2] + parameters[1]);// ( TP + TN ) / ( P + N )

	if ((performance_flags & retina::F_SCORE) == retina::F_SCORE)
	{
		float precision = (float)parameters[0] / (parameters[0] + parameters[2]);
		float recall = (float)parameters[0] / (parameters[0] + parameters[3]);

		result[4] = 2 * (precision * recall) / (precision + recall);
	}

	return result;
}

namespace retina
{
	void countTpTnFpFn(const cv::Mat& gold_standard, const cv::Mat& extracted_segment, std::vector<int>& tpTnFpFn)
	{

		if (gold_standard.size() != extracted_segment.size() || gold_standard.type() != CV_8U || extracted_segment.type() != CV_8U ||
			gold_standard.channels() != 1 || extracted_segment.channels() != 1)
		{

			std::cerr << "old_standard.size() != input.size() || gold_standard.type() != CV_8U || input.type() != CV_8U ||"
				"gold_standard.channels() != 1 || input.channels() != 1" << std::endl;

			return;
		}

		if (tpTnFpFn.size() < 4)
		{
			tpTnFpFn.clear();
			tpTnFpFn = { 0, 0, 0, 0 };
		}

		for (int i = 0; i < gold_standard.rows; i++)
		{
			for (int j = 0; j < gold_standard.cols; j++)
			{
				if (gold_standard.at< uchar >(i, j) > 100 && extracted_segment.at< uchar >(i, j) > 100)
				{
					tpTnFpFn[0] ++;//TP
				}
				if (gold_standard.at< uchar >(i, j) < 100 && extracted_segment.at< uchar >(i, j) < 100)
				{
					tpTnFpFn[1] ++;//TN
				}
				if (gold_standard.at< uchar >(i, j) > 100 && extracted_segment.at< uchar >(i, j) < 100)
				{
					tpTnFpFn[3] ++;//FN
				}
				if (gold_standard.at< uchar >(i, j) < 100 && extracted_segment.at< uchar >(i, j) > 100)
				{
					tpTnFpFn[2] ++;//FP
				}
			}
		}
	}
	std::vector<float> countPerformanceParameters(const cv::Mat& gold_standard, const cv::Mat& extracted_segment,
		int performance_flags, bool showTpTnFpFnPN)
	{

		if (gold_standard.size() != extracted_segment.size() || gold_standard.type() != CV_8U || extracted_segment.type() != CV_8U ||
			gold_standard.channels() != 1 || extracted_segment.channels() != 1)
		{

			std::cerr << "gold_standard.size() != input.size() || gold_standard.type() != CV_8U || input.type() != CV_8U ||"
				"gold_standard.channels() != 1 || input.channels() != 1" << std::endl;

			return std::vector<float>();
		}
		std::vector<int> parameters(4);
		parameters = { 0, 0, 0, 0};
		countTpTnFpFn(gold_standard, extracted_segment, parameters);

		std::vector<float> result = params2performanceParams(parameters, performance_flags);

		if (showTpTnFpFnPN)
		{
			std::cout << "TP = " << parameters[0] << ", TN = " << parameters[1] << ", FP = " << parameters[2] <<
				", FN = " << parameters[3] << ", P = " << parameters[0] + parameters[3] <<
				", N = " << parameters[2] + parameters[1] << std::endl;
		}

		return result;

	}
	std::vector<float> TpTnFpFn2PerformanceParameters(std::vector<int>& tpTnFpFn, int performance_flags, bool showTpTnFpFnPN)
	{

		if (tpTnFpFn.size() != 4)
		{
			std::cerr << "parameters.size() != 4" << std::endl;
			return std::vector<float>();
		}

		std::vector<float> result = params2performanceParams(tpTnFpFn, performance_flags);


		if (showTpTnFpFnPN)
		{
			std::cout << "TP = " << tpTnFpFn[0] << ", TN = " << tpTnFpFn[1] << ", FP = " << tpTnFpFn[2] <<
				", FN = " << tpTnFpFn[3] << ", P = " << tpTnFpFn[0] + tpTnFpFn[3] <<
				", N = " << tpTnFpFn[2] + tpTnFpFn[1] << std::endl;
		}

		return result;
	}

	void filterByLength(const cv::Mat & input, cv::Mat & output, unsigned int length)
	{

		if (input.channels() != 1 || input.type() != CV_8U)
		{
			std::cerr << "input.channels() != 1 || input.type() != CV_8U" << std::endl;
			return;
		}

		cv::Mat copyInput;
		input.copyTo(copyInput);
		output = cv::Mat::zeros(input.size(), input.type());


		for (int i = 0; i < copyInput.rows; i++)
		{
			for (int j = 0; j < copyInput.cols; j++)
			{

				if (copyInput.at< uchar >(i, j) == 255)
				{

					std::queue< cv::Point2i > coords;
					std::vector< cv::Point2i > outputCoords;

					copyInput.at< uchar >(i, j) = 100;

					coords.push(cv::Point2i(j, i));
					outputCoords.push_back(cv::Point2i(j, i));
					cv::Point2i currentCenter;

					while (!coords.empty())
					{

						currentCenter = coords.front();
						coords.pop();

						cv::Point2i currentPoint;

						currentPoint.x = currentCenter.x - 1;
						currentPoint.y = currentCenter.y - 1;

						while (currentPoint.y - currentCenter.y < 2)
						{

							if (currentPoint == currentCenter) currentPoint.x++;

							if (currentPoint.x >= 0 && currentPoint.x < copyInput.cols && currentPoint.y >= 0 &&
								currentPoint.y < copyInput.rows && (copyInput.at<uchar>(currentPoint.y, currentPoint.x) == 255))
							{
								copyInput.at< uchar >(currentPoint.y, currentPoint.x) = 100;
								coords.push(cv::Point2i(currentPoint.x, currentPoint.y));
								outputCoords.push_back(cv::Point2i(currentPoint.x, currentPoint.y));
							}

							currentPoint.x++;

							if (currentPoint.x - currentCenter.x >= 2)
							{
								currentPoint.y++;
								currentPoint.x = currentCenter.x - 1;
							}
						}
					}

					if (outputCoords.size() >= length)
					{

						for (unsigned int i = 0; i < outputCoords.size(); i++)
						{
							output.at< uchar >(outputCoords[i].y, outputCoords[i].x) = 255;
						}
					}

				}
			}
		}
	}

	void vessSegmentation(const cv::Mat& input, cv::Mat& output, bool specificity_param, bool show)
	{

		int width = input.cols;
		int height = input.rows;

		cv::Mat padded;

		cv::copyMakeBorder(input, padded, 30, 30, 30, 30, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		cv::Mat mask;

		if (!specificity_param)
			backgroundExclusion(padded, output, mask);
		else
			backgroundExclusion(padded, output, mask, 2.5, cv::Size(8, 8), 5, cv::Size(8, 8));

		threshold(output, output, 1, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		medianBlur(output, output, 3);

		if (!specificity_param)
			filterByLength(output, output, 150);
		else
			filterByLength(output, output, 300);

		cv::Rect rect(30, 30, width, height);
		output = output(rect);

		if (show)
		{
			cv::imshow("input", input);
			cv::imshow("output", output);
			cv::waitKey();
		}
	}
	void vessSegmentationGabor(const cv::Mat& input, cv::Mat& output, bool specificity_param, float sensitivity_param, bool show)
	{

		int width = input.cols;
		int height = input.rows;

		cv::Mat padded;

		cv::copyMakeBorder(input, padded, 30, 30, 30, 30, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		cv::Mat mask;

		if (!specificity_param)
			backgroundExclusion(padded, output, mask);
		else
			backgroundExclusion(padded, output, mask, 2.5, cv::Size(8, 8), 5, cv::Size(8, 8));

		GaborFilter4vessels(output, output);

		mask = 255 - mask;
		bool showHist = false;

		int thresholdValue = getThresholdUsingHist(output, mask, sensitivity_param, showHist);

		threshold(output, output, thresholdValue, 255, cv::THRESH_TRUNC);

		threshold(output, output, 0, 255, cv::THRESH_BINARY_INV | CV_THRESH_OTSU);

		cv::Rect rect(30, 30, width, height);
		output = output(rect);

		if (show)
		{
			cv::imshow("input", input);
			cv::imshow("output", output);
			cv::waitKey();
		}

	}
	
	void vesselsThiningMORPH(const cv::Mat& input, cv::Mat& output)
	{

		if (input.channels() != 1 || input.type() != CV_8U)
		{
			std::cerr << "input.channels() != 1 || input.type() != CV_8U" << std::endl;
			return;
		}

		input.copyTo(output);

		cv::Mat skel(input.size(), CV_8UC1, cv::Scalar(0));
		cv::Mat temp;
		cv::Mat eroded;

		cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

		bool done;
		do
		{
			cv::erode(output, eroded, element);
			cv::dilate(eroded, temp, element);
			cv::subtract(output, temp, temp);
			cv::bitwise_or(skel, temp, skel);
			eroded.copyTo(output);

			done = (cv::countNonZero(output) == 0);

		} while (!done);

		skel.copyTo(output);
	}

	cv::Point2i opticDiscLocalization(const cv::Mat& input, cv::Mat& output, bool show)
	{
		if (input.channels() > 1)
			splitColorImg(input, output, second);
		else
			input.copyTo(output);

		cv::Mat mask;
		threshold(output, mask, 40, 255, CV_THRESH_BINARY_INV);

		int morph_size = 3;
		cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
			cv::Point(morph_size, morph_size));

		morphologyEx(mask, mask, cv::MORPH_OPEN, element);

		morph_size = 30;
		element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
			cv::Point(morph_size, morph_size));

		morphologyEx(mask, mask, cv::MORPH_DILATE, element);
		morphologyEx(mask, mask, cv::MORPH_DILATE, element);

		output -= mask;

		cv::copyMakeBorder(output, output, 30, 30, 30, 30, cv::BORDER_CONSTANT, cv::Scalar::all(255));

		medianBlur(output, output, 25);

		cv::Rect rect(30, 30, input.cols, input.rows);
		output = output(rect);

		uchar max = 0;

		int indent = 100;
		for (int i = indent; i < output.rows - indent; i++)
		{
			for (int j = indent; j < output.cols - indent; j++)
			{
				if (output.at<uchar>(i, j) > max)
					max = output.at<uchar>(i, j);
			}
		}

		float number = 0, x = 0, y = 0;

		for (int i = indent; i < output.rows - indent; i++)
		{
			for (int j = indent; j < output.cols - indent; j++)
			{
				if (abs(output.at<uchar>(i, j) - max) < 1)
				{
					number++;
					x += j;
					y += i;
				}
			}
		}

		x /= number;
		y /= number;

		cv::Point2i center(cvRound(x), cvRound(y));

		if (show)
		{
			cv::Mat tmp;
			input.copyTo(tmp);

			int radius = 64;

			cv::circle(tmp, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
			cv::circle(tmp, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);

			cv::imshow("output", output);
			cv::imshow("result", tmp);
			cv::waitKey();
		}

		return center;
	}

	void opticDiscSegmentationRIM(const cv::Mat& input, cv::Mat& output, bool show)
	{

		input.copyTo(output);

		if (input.channels() > 1)
		{

			cvtColor(output, output, CV_BGR2HLS);

			splitColorImg(output, output, second);
		}

		int morph_size = 7;
		cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
			cv::Point(morph_size, morph_size));

		morphologyEx(output, output, cv::MORPH_CLOSE, element);

		cv::Mat tmp;

		morphologyEx(output, tmp, cv::MORPH_GRADIENT, element);

		output += tmp;

		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1);
		clahe->apply(output,output);

		blur(output, output, cv::Size(10, 10));

		threshold(output, output, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		filterByLength(output, output, 300);

		output = 255 - output;

		filterByLength(output, output, 700);

		output = 255 - output;

		morphologyEx(output, output, cv::MORPH_CLOSE, element);
		morphologyEx(output, output, cv::MORPH_ERODE, element);

		if (show)
		{
			imshow("input", input);
			imshow("output", output);
			cv::waitKey();
		}

	}
	void opticDiscSegmentation(const cv::Mat& input, cv::Mat& output, bool show)
	{

		cv::Point2i center = opticDiscLocalization(input, cv::Mat());

		float width = 128, height = 128;
		float x = center.x - width / 2;
		float y = center.y - height / 2;

		if (x < 0)
		{
			width += x;
			x = 0;
		}
		else if (x + width >= input.cols)
		{
			width = input.cols - 1 - x;
		}

		if (y < 0)
		{
			height += y;
			y = 0;
		}
		else if (y + height >= input.rows)
		{
			height = input.rows - 1 - y;
		}

		cv::Rect rect(x, y, width, height);
		cv::Mat roi = input(rect);

		cv::Mat opticDisc;
		opticDiscSegmentationRIM(roi, opticDisc);

		output = cv::Mat::zeros(input.size(), CV_8UC1);

		opticDisc.copyTo(output(rect));

		if (show)
		{
			cv::Mat tmp;

			if (input.channels() > 1) cvtColor(input, tmp, CV_BGR2GRAY);
			else input.copyTo(tmp);

			tmp += output;

			imshow("input", input);
			imshow("output", tmp);
			cv::waitKey();
		}

	}
}//end of namespace retina
