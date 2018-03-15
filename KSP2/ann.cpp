#include "stdafx.h"
#include "ann.h"
#include <cblas.h>

ANN::ANN(std::vector<int>& layerSize, int mode, int batchIfTrain)
{
	initDefault();
	setModeAndBatch(mode, batchIfTrain);
	setLayerSize(layerSize);
	allocate();
	if (mode == TRAIN || mode == FINETUNE) {
		reset();
		loadSamplePathAndLabels();
		shuffle();
	}
	cout << "finish init" << endl;
}

ANN::~ANN()
{
	release();
}

void ANN::setLayerSize(std::vector<int>& layerSize)
{
	numLayer = layerSize.size();
	cout << "Layer number: " << numLayer << endl;
	nLayers = new int[numLayer];
	for (int i = 0; i<numLayer; ++i)
		nLayers[i] = layerSize[i];
	wLayers.resize(numLayer - 1);
	bLayers.resize(numLayer);
	yLayers.resize(numLayer);
	if (mode != TEST) {
		gWLayers.resize(numLayer - 1);
		gBLayers.resize(numLayer);
		dLayers.resize(numLayer);
	}
}

void ANN::train(int iters)
{
	int iter = 0;
	int step_duration = 0;
	if (step>100)
		step_duration = step;
	else
		step_duration = max(1, iters / step);
	cout << "start train" << endl;
	while (iter<iters) {
		cv::Rect rect;
		for (int k = curIndex; k<curIndex + min(minibatch, numSamples - curIndex); ++k) {
			cv::Mat image = cv::imread(sampleFiles[ind[k]]);
			getFeature(image, k - curIndex);
			batchLabels[k - curIndex] = labelMat[ind[k]];
		}
		double loss = forward();
		backward();
		if (iter%display == 0 || iter == iters - 1)
			cout << "iteration: " << iter << ", loss: " << loss << endl;
		curIndex += minibatch;
		if (curIndex >= numSamples) {
			shuffle();
			curIndex = 0;
		}
		++iter;
		if (iter%step_duration == 0) {
			lr *= gamma;
		}
	}
	cout << endl << "finished" << endl;
}

void ANN::predict(cv::Mat & image, int & pred, float & prob)
{
	int batch = 1;
	getFeature(image, 0);
	for (int i = 0; i<numLayer - 1; ++i) {
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, nLayers[i + 1], nLayers[i], \
			1.0, yLayers[i], nLayers[i], wLayers[i], nLayers[i + 1], 0.0, yLayers[i + 1], nLayers[i + 1]);
		for (int j = 0; j<minibatch*nLayers[i + 1]; ++j) {
			//yLayers[i+1][j]+=bLayers[i+1][j%nLayers[i+1]];
			if (yLayers[i + 1][j]<0.0)
				yLayers[i + 1][j] = 0.0;
		}
	}
	double max_out = -1.0*1e3;
	double sum_out = 0.0;
	for (int j = 0; j<nLayers[numLayer - 1]; ++j) {
		if (yLayers[numLayer - 1][j]>max_out) {
			max_out = yLayers[numLayer - 1][j];
			pred = j;
		}
	}
	for (int j = 0; j<nLayers[numLayer - 1]; ++j) {
		output[j] = std::exp(yLayers[numLayer - 1][j] - max_out);
		sum_out += output[j];
	}
	for (int j = 0; j<nLayers[numLayer - 1]; ++j) {
		output[j] /= sum_out;
	}
	prob = output[pred];
	if (prob < 0.6)
		pred = -1;
}

void ANN::shuffle()
{
	for (size_t i = 0; i<ind.size(); ++i) {
		int swap_ind = rand() % (ind.size());
		swap(ind[i], ind[swap_ind]);
	}
}

float ANN::forward()
{
	int offset = 0;
	double sum_loss = 0.0;
	for (int i = 0; i<numLayer - 1; ++i) {
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, minibatch, nLayers[i + 1], nLayers[i], \
			1.0, yLayers[i], nLayers[i], wLayers[i], nLayers[i + 1], 0.0, yLayers[i + 1], nLayers[i + 1]);
		for (int j = 0; j<minibatch*nLayers[i + 1]; ++j) {
			yLayers[i+1][j]+=bLayers[i+1][j%nLayers[i+1]];
			if (yLayers[i + 1][j]<0.0)
				yLayers[i + 1][j] = 0.0;
		}
	}
	offset = 0;
	for (int i = 0; i<minibatch; ++i) {
		double max_out = -1.0*1e3;
		double sum_out = 0.0;
		for (int j = 0; j<nLayers[numLayer - 1]; ++j) {
			if (yLayers[numLayer - 1][j + offset]>max_out)
				max_out = yLayers[numLayer - 1][j + offset];
		}
		for (int j = 0; j<nLayers[numLayer - 1]; ++j) {
			output[j + offset] = std::exp(yLayers[numLayer - 1][j + offset] - max_out);
			sum_out += output[j + offset];
		}
		for (int j = 0; j<nLayers[numLayer - 1]; ++j) {
			output[j + offset] /= sum_out;
		}
		sum_loss += -1.0*std::log(output[offset + batchLabels[i]]);
		output[offset + batchLabels[i]] -= 1.0;
		for (int j = 0; j<nLayers[numLayer - 1]; ++j) {
			dLayers[numLayer - 1][j + offset] = output[j + offset] / minibatch;
		}
		offset += nLayers[numLayer - 1];
	}
	for (int i = numLayer - 1; i >= 1; --i) {
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nLayers[i - 1], nLayers[i], minibatch, \
			1.0, yLayers[i - 1], nLayers[i - 1], dLayers[i], nLayers[i], 0.0, gWLayers[i - 1], nLayers[i]);
		//dLayer1=dLayer2*W2T: [N*H]=[N*O].[O*H]
		if (i >= 2) {
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, minibatch, nLayers[i - 1], nLayers[i], \
				1.0, dLayers[i], nLayers[i], wLayers[i - 1], nLayers[i], 0.0, dLayers[i - 1], nLayers[i - 1]);
			for (int j = 0; j<minibatch*nLayers[i - 1]; ++j) {
				if (yLayers[i - 1][j]<1e-9)
					dLayers[i - 1][j] = 0.0;
			}
		}
		for (int j = 0; j<nLayers[i]; ++j) {
			gBLayers[i][j] = 0.0;
			for (int k = 0; k<minibatch; ++k) {
				gBLayers[i][j] += dLayers[i][j + k*nLayers[i]];
			}
		}
	}
	sum_loss /= minibatch;
	return sum_loss;
}

void ANN::backward()
{
	for (int i = 0; i<numLayer; ++i) {
		if (i<numLayer - 1) {
			for (int j = 0; j<nLayers[i] * nLayers[i + 1]; ++j)
				wLayers[i][j] -= (lr*gWLayers[i][j] + reg*wLayers[i][j]);
		}
		if (i >= 1) {
			for (int j = 0; j<nLayers[i]; ++j)
				bLayers[i][j] -= lr*gBLayers[i][j];
		}
	}
}

void ANN::reset(float sigma)
{
	std::mt19937 gen(time(0));
	std::normal_distribution<double> n(0, 1);
	int i, j;
	for (i = 0; i<numLayer; ++i) {
		if (i<numLayer - 1)
			for (j = 0; j<nLayers[i] * nLayers[i + 1]; ++j)
				wLayers[i][j] = sigma*n(gen);
		if (i >= 1)
			for (j = 0; j<nLayers[i]; ++j)
				bLayers[i][j] = 0.0;
	}
}

void ANN::initDefault()
{
	numLayer = -1;
	mode = TRAIN;
	minibatch = 32;
	curIndex = 0;
	step = 2;
	display = 10;
	gamma = 1.0;
	lr = 0.1;
	reg = 1e-4;
	sample_file_lst = "./label_file.txt";
}

void ANN::setModeAndBatch(int mode_, int batchIfTrain)
{
	mode = mode_;
	if (mode == TRAIN || mode == FINETUNE) {
		minibatch = batchIfTrain;
	}
	else
		minibatch = 1;
}

void ANN::allocate()
{
	cout << "batch size: " << minibatch << endl;
	bLayers[0] = nullptr;
	if (mode != TEST) {
		dLayers[0] = nullptr;
		gBLayers[0] = nullptr;
	}
	for (int i = 0; i<numLayer; ++i) {
		if (i<numLayer - 1) {
			wLayers[i] = new float[nLayers[i] * nLayers[i + 1]];
			if (mode != TEST)
				gWLayers[i] = new float[nLayers[i] * nLayers[i + 1]];
		}
		yLayers[i] = new float[minibatch*nLayers[i]];
		if (i >= 1) {
			bLayers[i] = new float[nLayers[i]];
			if (mode != TEST) {
				gBLayers[i] = new float[nLayers[i]];
				dLayers[i] = new float[minibatch*nLayers[i]];
			}
		}
	}
	output = new float[minibatch*nLayers[numLayer - 1]];
	if (mode != TEST)
		batchLabels = new int[minibatch];
}

void ANN::release()
{
	if (mode != TEST)
		delete batchLabels;
	delete output;
	for (int i = 0; i<numLayer; ++i) {
		if (i<numLayer - 1) {
			delete wLayers[i];
			if (mode != TEST) {
				delete gWLayers[i];
			}
		}
		delete yLayers[i];
		if (i >= 1) {
			delete bLayers[i];
			if (mode != TEST) {
				delete dLayers[i];
				delete gBLayers[i];
			}
		}
	}
}

void ANN::getFeature(cv::Mat & im, int batchIndex)
{
	int image_row = 80;
	int image_col = 80;
	int offset = nLayers[0];
	ptrExtractor->image_row_=image_row;
	ptrExtractor->image_col_ = image_col;
	ptrExtractor->extract(im, yLayers[0] + batchIndex*offset);
}

void ANN::preprocess(cv::Mat & image)
{
	//NOT_IMPLEMENTED
}

void ANN::loadSamplePathAndLabels()
{
	ifstream in;
	in.open(sample_file_lst.c_str(), ios::in);
	string line;
	int cnt = 0;
	cout << "loading samples" << endl;
	ifstream item_in;
	while (!in.eof()) {
		std::getline(in, line);
		if (line.length()<5)	continue;
		item_in.close();
		int slice = line.find_last_of(' ');		
		std::string filename = line.substr(0, slice);
		item_in.open(filename.c_str());
		if (!item_in) {
			cout << "file " << line << " not exists" << endl;
			continue;
		}

		int index = int(line[slice+1]) - 48;
		if (slice+1<line.length()) {
			char num[2] = { line[slice+1],line[slice + 2] };
			index = atoi(num);
		}
		sampleFiles.push_back(filename);
		labelMat.push_back(index);
		//if(index==10)
		//	cout<<line<<": "<<index<<endl;
		++cnt;
	}
	numSamples = cnt;
	for (int i = 0; i<numSamples; ++i) {
		ind.push_back(i);
	}
	cout << numSamples << " samples are loaded" << endl;
}

void ANN::saveParams(const char * filename)
{
	ofstream out;
	out.open(filename, ios::out);
	for (int i = 0; i<numLayer; ++i) {
		out << nLayers[i];
		if (i<numLayer - 1)
			out << ' ';
		else
			out << endl;
	}
	for (int i = 0; i<numLayer - 1; ++i) {
		out << "params" << endl;
		for (int j = 0; j<nLayers[i] * nLayers[i + 1]; ++j) {
			out << wLayers[i][j];
			if ((j + 1) % nLayers[i + 1] == 0)	out << endl;
			else 	out << ' ';
		}
	}
	for (int i = 1; i<numLayer; ++i) {
		out << "bias" << endl;
		for (int j = 0; j<nLayers[i]; ++j) {
			out << bLayers[i][j];
			if (j<nLayers[i] - 1)	out << ' ';
			else 	out << endl;
		}
	}

}

void ANN::loadParams(const char * filename)
{
	cout << "loading params..." << endl;
	ifstream in;
	in.open(filename, ios::in);
	if (!in) {
		cerr << "no parameter file found, reset randomly" << endl;
		reset();
	}
	string line;
	int layerIndexParams = -1;
	int layerIndexBias = 0;
	int paramRow = 0;

	int n;
	getline(in, line);
	stringstream ss(line);
	vector<int> layers;
	while (!ss.eof()) {
		ss >> n;
		layers.push_back(n);
	}
	setLayerSize(layers);
	allocate();
	while (!in.eof()) {
		getline(in, line);
		if (line.length() <= 1)	continue;
		if (line == "params") {
			++layerIndexParams;
			paramRow = 0;
		}
		else if (line == "bias") {
			++layerIndexBias;
		}
		else if (layerIndexBias == 0) {
			stringstream ss(line);
			double param;
			for (int i = 0; i<nLayers[layerIndexParams + 1]; ++i) {
				ss >> param;
				wLayers[layerIndexParams][paramRow*nLayers[layerIndexParams + 1] + i] = param;
			}
			++paramRow;
		}
		else if (layerIndexBias >= 1) {
			stringstream ss(line);
			double param;
			for (int i = 0; i<nLayers[layerIndexBias]; ++i) {
				ss >> param;
				bLayers[layerIndexBias][i] = param;
			}
		}
	}
	cout << "params loaded" << endl;
}
