// authentication_based_on_retinal_images.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "segmentation.h"
#include "similarity.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#pragma comment(lib, "opencv_core2410.lib")
#pragma comment(lib, "opencv_highgui2410.lib")
#pragma comment(lib, "opencv_imgproc2410.lib")

//////////
#include <Windows.h.>
#pragma comment(lib, "winmm.lib")
/////////




using namespace cv;
using namespace std;



const string vessels_ = "vessels";
const string disc = "disc";
const string test_whole_varia = "whole_varia";
const string test = "test";

const string rim_database = "RIM";
const string drive_database = "DRIVE";
const string varia_database = "VARIA";
const string hrf_database = "HRF";
const string messidor_database = "MESSIDOR";

const string resize_ = "resize";//it's only for RIM

//through space
vector<string> splitString(const string input)
{

	stringstream ssin(input);

	vector <string> words;

	while (ssin.good())
	{
		string word;
		ssin >> word;
		words.push_back(word);
	}

	return words;
}


int main(int argc, char**argv)
{

	if (argv[1] == vessels_)
	{
		if (argv[2] == drive_database)
		{
			ifstream retinas_file(argv[3], ios_base::in);
			ifstream vessels_file(argv[4], ios_base::in);

			vector<float> counted_params;
			vector<int> parameters;

			if (vessels_file.is_open() && retinas_file.is_open())
			{

				string vessels_name, retina_name, mask_name;

				while (getline(vessels_file, vessels_name) && getline(retinas_file, retina_name))
				{

					string vessels_img_path = "C:\\DRIVE\\test\\1st_manual\\";
					vessels_img_path += vessels_name;

					string retina_img_path = "C:\\DRIVE\\test\\images\\";
					retina_img_path += retina_name;

					Mat retina = imread(retina_img_path, CV_LOAD_IMAGE_COLOR);

					Mat gold_standard = imread(vessels_img_path, CV_LOAD_IMAGE_GRAYSCALE);

					Mat extracted_vessels;

					//unsigned long t1, t2;
					//timeBeginPeriod(1);
					//t1 = timeGetTime();
					retina::vessSegmentationGabor(retina, extracted_vessels, true);
					//imshow("ves", extracted_vessels);
					//waitKey();
					//t2 = timeGetTime() - t1;
					//cout << "time : " << t2 << " milliseconds" << endl;

					retina::countTpTnFpFn(gold_standard, extracted_vessels, parameters);

				}

				vessels_file.close();
				retinas_file.close();

			}

			counted_params = retina::TpTnFpFn2PerformanceParameters(parameters, retina::PERFORMANCE_PARAMS::TPR | retina::PERFORMANCE_PARAMS::FPR |
				retina::PERFORMANCE_PARAMS::ACC | retina::PERFORMANCE_PARAMS::SPC, true);

			cout << "TPR = " << counted_params[0] << ", FPR = " << counted_params[1] << ", SPC = " <<
				counted_params[2] << ", ACC = " << counted_params[3] << endl;
		}
		else if (argv[2] == messidor_database)
		{
			ifstream retinas_file(argv[3], ios_base::in);

			if (retinas_file.is_open())
			{

				string retina_name;

				string retina_img_dir = "C:\\MESSIDOR\\";
				string retina_img_vess_dir = "C:\\MESSIDOR\\vessels\\";


				while (getline(retinas_file, retina_name))
				{

					string retina_img_path = retina_img_dir + retina_name;

					Mat retina = imread(retina_img_path, CV_LOAD_IMAGE_GRAYSCALE);

					resize(retina, retina, Size(896, 595));

					int indent = 160;
					Rect rect(indent, 0, retina.cols - 2 * indent, retina.rows);
					retina = retina(rect);

					Mat vessels;
					retina::vessSegmentation(retina, vessels, true);

					retina_img_path = retina_img_vess_dir + retina_name;

					imwrite(retina_img_path, vessels);
				}

				retinas_file.close();
			}
		}
	}
	else if (argv[1] == disc)
	{

		if (argv[2] == rim_database)
		{
			ifstream retinas_file(argv[3], ios_base::in);
			ifstream disc_file(argv[4], ios_base::in);

			vector<float> counted_params;
			vector<int> parameters;

			if (retinas_file.is_open() && disc_file.is_open())
			{

				string disc_name, retina_name;

				while (getline(retinas_file, retina_name) && getline(disc_file, disc_name))
				{
					string retina_img_path = "C:\\RIM-ONE_database_r1\\Normal\\";
					retina_img_path += retina_name;

					string disc_img_path = "C:\\RIM-ONE_database_r1\\Normal\\";
					disc_img_path += disc_name;

					Mat retina = imread(retina_img_path, CV_LOAD_IMAGE_COLOR);
					Mat gold_standard = imread(disc_img_path, CV_LOAD_IMAGE_GRAYSCALE);

					Mat extracted_disc;


					if (argc == 6)
					{
						if (argv[5] == resize_)
						{

							int width = 512;//128;
							int height = 512;//128;

							resize(retina, retina, cv::Size(width, height));
							resize(gold_standard, gold_standard, retina.size());

						}
					}

					unsigned long t1, t2;
					timeBeginPeriod(1);
					t1 = timeGetTime();

					retina::opticDiscSegmentationRIM(retina, extracted_disc, true);
					t2 = timeGetTime() - t1;

					cout << "time : " << t2 << " milliseconds" << endl;

					retina::countTpTnFpFn(gold_standard, extracted_disc, parameters);

				}

				retinas_file.close();
				disc_file.close();

			}

			counted_params = retina::TpTnFpFn2PerformanceParameters(parameters, retina::PERFORMANCE_PARAMS::TPR | retina::PERFORMANCE_PARAMS::FPR |
				retina::PERFORMANCE_PARAMS::ACC | retina::PERFORMANCE_PARAMS::SPC | retina::PERFORMANCE_PARAMS::F_SCORE);

			cout << "TPR = " << counted_params[0] << ", FPR = " << counted_params[1] << ", SPC = " <<
				counted_params[2] << ", ACC = " << counted_params[3] << ",\n\rF_SCORE = " << counted_params[4] << endl;

		}
		else if (argv[2] == drive_database)
		{

			ifstream retinas_file(argv[3], ios_base::in);

			if (retinas_file.is_open())
			{

				string retina_name;

				while (getline(retinas_file, retina_name))
				{
					string retina_img_path = "C:\\DRIVE\\test\\images\\";;
					retina_img_path += retina_name;

					Mat retina = imread(retina_img_path, CV_LOAD_IMAGE_COLOR);

					Mat extracted_disc;
					unsigned long t1, t2;
					timeBeginPeriod(1);
					t1 = timeGetTime();

					retina::opticDiscSegmentation(retina, extracted_disc, true);

					t2 = timeGetTime() - t1;
					cout << "time : " << t2 << " milliseconds" << endl;
				}

				retinas_file.close();

			}

		}
		else if (argv[2] == varia_database)
		{

			ifstream retinas_file(argv[3], ios_base::in);

			if (retinas_file.is_open())
			{

				string retina_name;

				while (getline(retinas_file, retina_name))
				{
					string retina_img_path = "C:\\VARIA\\";
					retina_img_path += retina_name;

					Mat retina = imread(retina_img_path, 0);

					resize(retina, retina, Size(retina.cols / 1.8, retina.rows / 1.8));

					cv::Mat opticDisc;
					cout << retina_img_path << endl;
					retina::opticDiscSegmentation(retina, opticDisc, true);

				}

				retinas_file.close();

			}

		}
	}
	else if (argv[1] == test_whole_varia)
	{

		ifstream retinas_file(argv[2], ios_base::in);
		vector<string> retinas_names;

		if (retinas_file.is_open())
		{
			string retina_name;

			while (getline(retinas_file, retina_name))
			{
				retinas_names.push_back(retina_name);
			}
			retinas_file.close();
		}

		ifstream index_file(argv[3], ios_base::in);
		vector<string> index;

		if (index_file.is_open())
		{
			string identical_retinas;

			while (getline(index_file, identical_retinas))
			{
				index.push_back(identical_retinas);
			}
			index_file.close();
		}

		const string retina_img_dir = "C:\\VARIA\\";

		vector<vector<string>> sim_retinas(retinas_names.size());
		vector<vector<int>> tptnfpfn(retinas_names.size());

		for (unsigned int i = 0; i < retinas_names.size(); i++)
		{
			for (int j = 0; j < 4; j++)
			{
				tptnfpfn[i].push_back(0);
			}
		}

		vector<Mat> vessels(retinas_names.size());

		for (unsigned int i = 0; i < retinas_names.size(); i++)
		{
			string retina_img_path = retina_img_dir + retinas_names[i];

			Mat retina = imread(retina_img_path, CV_LOAD_IMAGE_COLOR);

			resize(retina, retina, Size(retina.cols / 1.4, retina.rows / 1.4));

			cv::Mat vessels_m;
			retina::vessSegmentation(retina, vessels_m, true);
			vessels[i] = vessels_m;

			cout << i << endl;
		}

		float minNonSim = 1, minSim = 1;
		float maxNonSim = 0, maxSim = 0;

		retina::Retina* retina = new retina::PCRetina(3);

		const float threshold = 0.25f;

		for (int i = 0; i < retinas_names.size() - 1; i++)
		{
			sim_retinas[i].push_back(retinas_names[i]);
			vector<float> vecSim(retinas_names.size() - i - 1);

			unsigned int idx = 0;
			for (; idx < index.size(); idx++)
			{
				string name = retinas_names[i].substr(0, 4);
				size_t found = index[idx].find(name);

				if (found != std::string::npos)
				{
					break;
				}
			}


			for (int j = i + 1; j < retinas_names.size(); j++)
			{

				float sim = retina->compareVessels(vessels[i], vessels[j]);

				vecSim[j - (i + 1)] = sim;

				//cout << "i = " << i << "; j = " << j << " sim = " << sim << endl;

				if (sim >= threshold)
				{
					//cout << "j = " << j << " sim = " << sim << endl;
					sim_retinas[i].push_back(retinas_names[j]);
				}

			}

			if (idx == index.size())// similar retinas are absent
			{
				tptnfpfn[i][1] += retinas_names.size() - i - sim_retinas[i].size();
				tptnfpfn[i][2] += sim_retinas[i].size() - 1;

				for (unsigned int j = 0; j < vecSim.size(); j++)
				{
					if (vecSim[j] < minNonSim) minNonSim = vecSim[j];
					if (vecSim[j] > maxNonSim) maxNonSim = vecSim[j];
				}
			}
			else
			{
				vector<string> realSim = splitString(index[idx]);

				for (unsigned int ii = 0; ii < realSim.size(); ii++)
				{
					if (realSim[ii] == sim_retinas[i][0].substr(0, 4))
					{
						string tmp = realSim[0];
						realSim[0] = realSim[ii];
						realSim[ii] = tmp;
					}
				}

				int tpfpfn = 0;

				for (unsigned int ii = 1; ii < realSim.size(); ii++)
				{

					string tmp = realSim[ii].substr(1, 3);
					unsigned int iii = stoi(tmp) - 1;

					if (iii < i + 1) continue;

					if (vecSim[iii - i - 1] < minSim) minSim = vecSim[iii - i - 1];
					if (vecSim[iii - i - 1] > maxSim) maxSim = vecSim[iii - i - 1];

					vecSim[iii - i - 1] = -1;

					bool found = false;
					for (unsigned int j = 1; j < sim_retinas[i].size(); j++)
					{
						if (realSim[ii] == sim_retinas[i][j].substr(0, 4))
						{
							sim_retinas[i][j].at(0) = 'T';
							found = true;
							break;
						}
					}

					if (found)
					{
						tptnfpfn[i][0]++;
						tpfpfn++;
					}
					else
					{
						tptnfpfn[i][3]++;
						tpfpfn++;
					}
				}

				for (unsigned int j = 0; j < vecSim.size(); j++)
				{
					if (vecSim[j] != -1)
					{
						if (vecSim[j] < minNonSim) minNonSim = vecSim[j];
						if (vecSim[j] > maxNonSim) maxNonSim = vecSim[j];
					}
				}

				for (unsigned int j = 1; j < sim_retinas[i].size(); j++)
				{
					if (sim_retinas[i][j].at(0) != 'T')
					{
						tptnfpfn[i][2]++;
						tpfpfn++;
					}
				}

				tptnfpfn[i][1] = retinas_names.size() - i - tpfpfn - 1;

			}//end of else

			cout << "i = " << i << " : " << tptnfpfn[i][0] << " " << tptnfpfn[i][1] << " " << tptnfpfn[i][2] << " " << tptnfpfn[i][3] <<
				" minNonSim = " << minNonSim << " maxNonSim = " << maxNonSim << " minSim = " << minSim << " maxSim = " << maxSim << endl;
		}

		vector<int> params(4);
		params = { 0, 0, 0, 0 };

		for (unsigned int i = 0; i < retinas_names.size(); i++)
		{
			params[0] += tptnfpfn[i][0];
			params[1] += tptnfpfn[i][1];
			params[2] += tptnfpfn[i][2];
			params[3] += tptnfpfn[i][3];
		}

		vector<float> counted_params;
		counted_params = retina::TpTnFpFn2PerformanceParameters(params, retina::PERFORMANCE_PARAMS::TPR | retina::PERFORMANCE_PARAMS::FPR |
			retina::PERFORMANCE_PARAMS::SPC | retina::PERFORMANCE_PARAMS::ACC | retina::PERFORMANCE_PARAMS::F_SCORE, true);

		cout << "TPR = " << counted_params[0] << ", FPR = " << counted_params[1] << ", SPC = " <<
			counted_params[2] << ", ACC = " << counted_params[3] << endl;
	}
	else if (argv[1] == test)
	{
		if (argv[2] == varia_database)
		{
			ifstream index_file(argv[3], ios_base::in);
			vector<string> names_from_index;

			if (index_file.is_open())
			{
				string identical_retinas;

				while (getline(index_file, identical_retinas))
				{
					names_from_index.push_back(identical_retinas);
				}
				index_file.close();
			}

			vector<vector<int>> index(names_from_index.size());

			for (int i = 0; i < index.size(); i++)
			{
				vector<string> words = splitString(names_from_index[i]);

				for (int j = 0; j < words.size(); j++)
				{
					string tmp = words[j].substr(1, 3);
					unsigned int idx = stoi(tmp) - 1;
					index[i].push_back(idx);
				}
			}

			const string retina_img_dir = "C:\\VARIA\\R";
			const string format = ".pgm";

			float min = 1;

			for (int i = 0; i < index.size(); i++)
			{

				int start = 0;

				while (start < index[i].size() - 1)
				{
					string name;
					if (index[i][start] + 1 >= 100)
					{
						name = to_string(index[i][start] + 1);
					}
					else if (index[i][start] + 1 < 10)
					{
						name = "00" + to_string(index[i][start] + 1);
					}
					else
					{
						name = "0" + to_string(index[i][start] + 1);
					}

					string path1 = retina_img_dir + name + format;
					Mat retina1 = imread(path1, 1);

					resize(retina1, retina1, Size(retina1.cols / 1.4, retina1.rows / 1.4));

					cv::Mat vessels1;
					retina::vessSegmentation(retina1, vessels1, true);

					for (int j = start + 1; j < index[i].size(); j++)
					{
						string name;
						if (index[i][j] + 1 >= 100)
						{
							name = to_string(index[i][j] + 1);
						}
						else if (index[i][j] + 1 < 10)
						{
							name = "00" + to_string(index[i][j] + 1);
						}
						else
						{
							name = "0" + to_string(index[i][j] + 1);
						}

						string path2 = retina_img_dir + name + format;
						Mat retina2 = imread(path2, 1);

						resize(retina2, retina2, Size(retina2.cols / 1.4, retina2.rows / 1.4));

						cv::Mat vessels2;
						retina::vessSegmentation(retina2, vessels2, true);


						retina::Retina* retina = new retina::PCRetina(3);

						unsigned long t1, t2;
						timeBeginPeriod(1);
						t1 = timeGetTime();


						float sim = retina->compareVessels(vessels1, vessels2);
						t2 = timeGetTime() - t1;

						cout << "time : " << t2 << " milliseconds" << endl;

						if (sim < min) min = sim;

						cout << index[i][start] + 1 << ", " << index[i][j] + 1 << " : " << sim << " " << min << endl;

					}
					start++;
				}
			}
		}
		else if (argv[2] == hrf_database)
		{

			ifstream names_file(argv[3], ios_base::in);
			vector<string> names;

			if (names_file.is_open())
			{
				string retina_name;

				while (getline(names_file, retina_name))
				{
					names.push_back(retina_name);
				}
				names_file.close();
			}

			vector<Mat> vessels(names.size());

			const string retina_img_dir = "C:\\HRF\\";

			for (int i = 0; i < names.size(); i++)
			{
				string retina_img_path = retina_img_dir + names[i];

				Mat retina = imread(retina_img_path, CV_LOAD_IMAGE_GRAYSCALE);//sRGB is in original

				//cout << retina.cols / 5 << "  " << retina.rows / 5 << endl;
				resize(retina, retina, Size(777, 518));

				int indent = 120;
				Rect rect(indent, 0, retina.cols - 2 * indent, retina.rows);
				retina = retina(rect);

				cv::Mat vessels_;
				retina::vessSegmentation(retina, vessels_, true);

				//imshow("retina", retina);
				//imshow("ves", vessels_);
				//waitKey();

				vessels[i] = vessels_;

				cout << names[i] << endl;
			}

			const float threshold = 0.19;

			retina::Retina* retina = new retina::PCRetina(3);

			for (int i = 0; i < vessels.size(); i++)
			{
				for (int j = i + 1; j < vessels.size(); j++)
				{
					float sim = retina->compareVessels(vessels[i], vessels[j]);

					//imshow("i", vessels[i]);
					//imshow("j", vessels[j]);

					//cout << "j = " << names[j] << " , i = " << names[i] << " : sim = " << sim << endl;

					//waitKey();

					if (sim >= threshold)
					{
						//cout << "###" << endl;
						cout << "j = " << names[j] << " , i = " << names[i] << " : sim = " << sim << endl;
					}
				}
			}
		}
		else if (argv[2] == messidor_database)
		{

			ifstream names_file(argv[3], ios_base::in);
			vector<string> names;

			if (names_file.is_open())
			{
				string retina_name;

				while (getline(names_file, retina_name))
				{
					names.push_back(retina_name);
				}
				names_file.close();
			}

			const float threshold = 0.19f;

			const string retina_img_dir = "C:\\MESSIDOR\\vessels\\";

			retina::Retina* retina = new retina::PCRetina(3);

			for (int i = stoi(argv[4]); i >= 0; i--)//140 zlat; 705 chel вкл. их
			{

				cout << "i  = " << i << endl;

				string img_path1 = retina_img_dir + names[i];
				Mat vessels1 = imread(img_path1, 0);

				for (int j = i + 1; j < names.size(); j++)
				{

					string img_path2 = retina_img_dir + names[j];
					Mat vessels2 = imread(img_path2, 0);

					float sim = retina->compareVessels(vessels1, vessels2);

					if (sim >= threshold)
					{

						cout << i << " j = " << names[j] << " , i = " << names[i] << " : sim = " << sim << endl;

						ofstream results("C:\\MESSIDOR\\mess.txt", ios_base::in | ios_base::app);

						if (results.is_open())
						{
							results << i << " j = " << names[j] << " , i = " << names[i] << " : sim = " << sim << endl;
							results.close();
						}

					}

				}
			}
		}
	}

	return 0;
}

