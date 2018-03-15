#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <cmath>
#include <memory>
#include "featureExtractor.h"
using namespace std;

#define TRAIN			0
#define TEST			1
#define FINETUNE	2

class ANN
{
public:
	ANN(std::vector<int>& layerSize, int mode=TRAIN, int batchIfTrain=32);
	ANN(int mode_=TEST) {
		setModeAndBatch(mode, 1);
	}
	virtual ~ANN();

private:
	std::shared_ptr<FeatureExtractor> ptrExtractor;
public:
	int step;
	int display;
	float gamma;
	float lr;
	float reg;
	
	int* nLayers;
	int mode;
	int minibatch;
	int numLayer;
	float* output;
	float* batchSamples;
	int* batchLabels;
	vector<float*> wLayers;
	vector<float*> bLayers;
	vector<float*> gWLayers;
	vector<float*> gBLayers;
	vector<float*> yLayers;
	vector<float*> dLayers;
	
	vector<string> sampleFiles;
	vector<int> labelMat;
	vector<int> ind;

public:
	int curIndex;
	int numSamples;
	string sample_file_lst;

public:
	inline void setFeatureExtractorPtr(std::shared_ptr<FeatureExtractor>& ptr) {
		ptrExtractor = ptr;
	}
	void train(int iters = 1e6);
	void setLayerSize(std::vector<int>& layerSize);
	void predict(cv::Mat& image, int& pred, float& prob);
	void saveParams(const char* filename);
	void loadParams(const char* filename);

private:
	void shuffle();
	float forward();
	void backward();
	void reset(float sigma = 1e-2);
	void initDefault();
	void setModeAndBatch(int mode_, int batchIfTrain);
	void allocate();
	void release();
	void getFeature(cv::Mat& im, int batchIndex);
	void preprocess(cv::Mat& image);
	void loadSamplePathAndLabels();
};

