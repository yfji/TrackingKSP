#pragma once
#include <opencv2/opencv.hpp>
class FeatureExtractor
{
public:
	FeatureExtractor();
	virtual ~FeatureExtractor();

public:
	int image_row_;		//align
	int feat_dim_;		//feature dimension
	int image_col_;

	virtual float* extract(cv::Mat& image) = 0;
	virtual void extract(cv::Mat& image, float* feat) = 0;
};

