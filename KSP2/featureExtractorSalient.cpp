#include "stdafx.h"
#include "featureExtractorSalient.h"


FeatureExtractorSalient::FeatureExtractorSalient()
{
}


FeatureExtractorSalient::~FeatureExtractorSalient()
{
}

void FeatureExtractorSalient::calcHog(cv::Mat & image, float * hogFeat)
{
	//检测窗口(80,80),块尺寸(16,16),块步长(8,8),cell尺寸(16,16),直方图bin个数9
	cv::HOGDescriptor desc(cv::Size(image_col_, image_row_), cv::Size(16, 16), cv::Size(8, 8), cv::Size(16, 16), 9);
	cv::Mat sample;
	cv::resize(image, sample, cv::Size(image_col_, image_row_), cv::INTER_CUBIC);
	std::vector<float> hog_feat;
	desc.compute(sample, hog_feat, cv::Size(16, 16));  //检测窗口移动步长(16,16)
													   //std::cout << "feat dim: " << hog_feat.size() << std::endl;
	float _min = 1e5;
	float _max = -1e5;
	for (auto i = 0; i < hog_feat.size(); ++i) {
		hogFeat[i] = hog_feat[i];
		if (hog_feat[i] > _max)
			_max = hog_feat[i];
		if (hog_feat[i] < _min)
			_min = hog_feat[i];
	}
	_max -= _min;
	for (auto i = 0; i < hog_feat.size(); ++i) {
		hogFeat[i] = (hogFeat[i] - _min) / _max;
	}
}

void FeatureExtractorSalient::calcHist(cv::Mat & image, float * histFeat)
{
	const int binSize = 16;
	const int block_num = 4;
	int bin_num = 256 / binSize;
	int feat_len = bin_num*block_num*block_num;
	int pool_h = image.rows / block_num;
	int pool_w = image.cols / block_num;

	uchar* data = (uchar*)image.data;
	for (auto i = 0; i < feat_len; ++i)
		histFeat[i] = 0.0;
	float _min = 1e5;
	float _max = -1e5;
	for (auto i = 0; i < block_num; ++i) {
		for (auto j = 0; j < block_num; ++j) {

			calcHistPatch(data + i*block_num*pool_h + j*pool_w, \
				histFeat + (i*block_num + j)*bin_num, \
				binSize, \
				image.cols, pool_h, pool_w);
		}
	}
	//for (auto i = 0; i < image.rows*image.cols; ++i) {
	//	int bin_ind = (int)data[i] / binSize;
	//	histFeat[bin_ind]+=1;
	//	if (histFeat[bin_ind] > _max)
	//		_max = histFeat[bin_ind];
	//	if (histFeat[bin_ind] < _min)
	//		_min = histFeat[bin_ind];
	//}
	//for (auto i = 0; i < bin_num; ++i)
	//	histFeat[i] = (histFeat[i] - _min) / _max;
}

void FeatureExtractorSalient::calcHistPatch(const uchar * pData, float* feat, int binSize, int offset, int h, int w)
{
	float _min = 1e5;
	float _max = -1e5;
	int bin_num = 256 / binSize;
	for (auto i = 0; i < h; ++i) {
		for (auto j = 0; j < w; ++j) {
			int bin_ind = (int)pData[i*offset + j] / binSize;
			feat[bin_ind] += 1;
			if (feat[bin_ind] > _max)
				_max = feat[bin_ind];
			if (feat[bin_ind] < _min)
				_min = feat[bin_ind];

		}
	}
	_max -= _min;
	for (auto i = 0; i < bin_num; ++i)
		feat[i] = (feat[i] - _min) / _max;
}

float * FeatureExtractorSalient::extract(cv::Mat & image)
{
	return nullptr;
}

void FeatureExtractorSalient::extract(cv::Mat & image, float * feat)
{
	cv::Mat temp = image;
	if (image.channels() == 3)
		cv::cvtColor(image, temp, cv::COLOR_BGR2GRAY);
	calcHog(temp, feat);
	//calcHist(temp, feat + 729);
}
