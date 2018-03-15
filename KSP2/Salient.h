#pragma once
#include <opencv2\saliency.hpp>
#include "ann.h"
#include "featureExtractorSalient.h"

class Salient
{
public:
	Salient();
	virtual ~Salient();

	cv::Mat salientDetectFT(cv::Mat& im);
	inline cv::Mat binarize(cv::Mat& im) {
		cv::Mat biMap;
		cv::threshold(im, biMap, 50, 255, cv::THRESH_OTSU);
		return biMap;
	}
	cv::Mat adaptBinarize(cv::Mat& im);
	std::vector<std::pair<cv::Rect, float>> findBoundingBoxes(const cv::Mat& im);
	std::shared_ptr<FeatureExtractor> ptrExtractor;
private:
	const int times = 2;
	int max_area;
	int min_area;
	
	std::shared_ptr<ANN> nn;

private:
	inline float gamma(float x) {
		return x>0.04045 ? pow((x + 0.055f) / 1.055f, 2.4f) : x / 12.92;
	}
	
	void RGBToLab(unsigned char * rgbImg, float * labImg);
	
};

