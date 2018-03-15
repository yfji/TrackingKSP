#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "Salient.h"
#include "featureExtractor.h"
#include "featureExtractorSalient.h"
#include "ann.h"
#include "reader.h"
using namespace std;
using namespace cv;

#define MAX_T	20

enum TargetType {
	person = 0,
	vehicle,
	background
};

struct target {
	cv::Rect box;
	int frameId;
	int trueId;
	int featLen;
	float confidence;
	bool searched;
	TargetType type;
	float* feature;
	target() :box(), frameId(0), trueId(0), confidence(0.0), searched(false), type(background) {}
	target(const target& t) {
		box = t.box;
		frameId = t.frameId;
		trueId = t.trueId;
		featLen = t.featLen;
		feature = t.feature;
		confidence = t.confidence;
		searched = t.searched;
		type = t.type;
	}
};
class Detection
{
public:
	Detection();
	~Detection();

private:
	Salient salient;
	FeatureExtractorSalient ex;
	const int featLen = 256;
	int totalSize;
	std::string filename;
	vector<float*> features;
	vector<vector<int>> disjointPaths;
	std::vector<std::string> file_paths;

	void detect(cv::Mat& image, vector<target>& frame_detections, int offset, int scale = 2);
	void extract(cv::Mat& roi, float* feat);

	const cv::Scalar colors[MAX_T] = {
		cv::Scalar(0,0,255),
		cv::Scalar(0,255,255),
		cv::Scalar(0,255,0),
		cv::Scalar(255,128,128),
		cv::Scalar(192,128,255),
		cv::Scalar(255,255,128),
		cv::Scalar(128,128,0),
		cv::Scalar(0,128,128),
		cv::Scalar(64,128,255),
		cv::Scalar(192,192,192),
		cv::Scalar(255,128,0),
		cv::Scalar(64,128,0),
		cv::Scalar(64,0,64),
		cv::Scalar(0,0,128),
		cv::Scalar(192,128,128),
		cv::Scalar(255,0,255),
		cv::Scalar(64,128,128),
		cv::Scalar(0,128,0),
		cv::Scalar(0,0,0),
		cv::Scalar(255,255,255)
	};

public:
	target** nodes;
	int* prev;
	int* next;
	int* ids;
	vector<int> frameSizes;
	vector<vector<target>> detections;

	void runVideo(const char* filename);
	void parseDisjointPaths();

	void drawBoundingBoxes();
};

