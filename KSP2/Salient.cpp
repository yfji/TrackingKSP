#include "stdafx.h"
#include "Salient.h"

#define CUT_BORDER	1

Salient::Salient()
{
	ptrExtractor = std::make_shared<FeatureExtractorSalient>();
	nn = std::make_shared<ANN>();
	nn->loadParams("./model/param_hog_iter_10000.model");
	nn->setFeatureExtractorPtr(ptrExtractor);
}

Salient::~Salient()
{
}

void Salient::RGBToLab(unsigned char * rgbImg, float * labImg) {
	float B = gamma(rgbImg[0] / 255.0f);
	float G = gamma(rgbImg[1] / 255.0f);
	float R = gamma(rgbImg[2] / 255.0f);
	float X = 0.412453*R + 0.357580*G + 0.180423*B;
	float Y = 0.212671*R + 0.715160*G + 0.072169*B;
	float Z = 0.019334*R + 0.119193*G + 0.950227*B;

	X /= 0.95047;
	Y /= 1.0;
	Z /= 1.08883;

	float FX = X > 0.008856f ? pow(X, 1.0f / 3.0f) : (7.787f * X + 0.137931f);
	float FY = Y > 0.008856f ? pow(Y, 1.0f / 3.0f) : (7.787f * Y + 0.137931f);
	float FZ = Z > 0.008856f ? pow(Z, 1.0f / 3.0f) : (7.787f * Z + 0.137931f);
	labImg[0] = Y > 0.008856f ? (116.0f * FY - 16.0f) : (903.3f * Y);
	labImg[1] = 500.f * (FX - FY);
	labImg[2] = 200.f * (FY - FZ);
}

cv::Mat Salient::salientDetectFT(cv::Mat& im) {
	assert(src.channels() == 3);
	cv::Mat salMap(im.size(), CV_32FC1);
	cv::Mat lab, labf;
	int h = im.rows, w = im.cols;
	labf.create(cv::Size(w, h), CV_32FC3);
	uchar* fSrc = im.data;
	float* fLab = (float*)labf.data;
	float* fDst = (float*)salMap.data;

	int stride = w * 3;
	//for (int i = 0; i < h; ++i) {
	//	for (int j = 0; j < stride; j += 3) {
	//		RGBToLab(fSrc + i*stride + j, fLab + i*stride + j);
	//	}
	//}
	float MeanL = 0, MeanA = 0, MeanB = 0;
	for (int i = 0; i < h; ++i) {
		int index = i*stride;
		for (int x = 0; x < w; ++x) {
			RGBToLab(fSrc + index, fLab + index);
			MeanL += fLab[index];
			MeanA += fLab[index + 1];
			MeanB += fLab[index + 2];
			index += 3;
		}
	}
	MeanL /= (w * h);
	MeanA /= (w * h);
	MeanB /= (w * h);
	cv::GaussianBlur(labf, labf, cv::Size(5, 5), 1);
	for (int Y = 0; Y < h; Y++)
	{
		int Index = Y * stride;
		int CurIndex = Y * w;
		for (int X = 0; X < w; X++)
		{
			fDst[CurIndex++] = (MeanL - fLab[Index]) *  \
				(MeanL - fLab[Index]) + (MeanA - fLab[Index + 1]) *  \
				(MeanA - fLab[Index + 1]) + (MeanB - fLab[Index + 2]) *  \
				(MeanB - fLab[Index + 2]);
			Index += 3;
		}
	}
	cv::normalize(salMap, salMap, 0, 1, cv::NORM_MINMAX);
	salMap.convertTo(salMap, CV_8UC1, 255);
	return salMap;
}

std::vector<std::pair<cv::Rect, float>> Salient::findBoundingBoxes(const cv::Mat& im){
	std::vector<std::pair<cv::Rect, float>> boxes;
	IplImage ipl = im;
	CvMemStorage* pStorage = cvCreateMemStorage(0);
	CvSeq* pContour = NULL;
	cvFindContours(&ipl, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	bool updated = false;
	max_area = 4e4;
	min_area = 200;
	bool use_nn = 1;
	for (; pContour; pContour = pContour->h_next) {
		float true_area = fabs(cvContourArea(pContour));
		cv::Rect bbox = cvBoundingRect(pContour, 0);
		float box_area = 1.0*bbox.height*bbox.width;
		if (bbox.width > 2*bbox.height || bbox.height > 2.5*bbox.width) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		if (bbox.width > im.cols / 2 || bbox.height > im.rows / 2) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		if (box_area > max_area || box_area < min_area) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		if (box_area / true_area > 4.1) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		int pred;
		float prob;
		int pad = 5;
		cv::Rect detbox = cv::Rect(max(0, bbox.x - pad), max(0, bbox.y - pad), bbox.width + 2*pad, bbox.height + 2*pad);
		detbox.width = min(im.cols - detbox.x, detbox.width);
		detbox.height = min(im.rows - detbox.y, detbox.height);
		nn->predict(im(detbox), pred, prob);
		if (pred != 1) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		//std::cout << "ratio: " << box_area / true_area << std::endl;
		boxes.push_back(std::make_pair(bbox,prob));
	}
	cvReleaseMemStorage(&pStorage);
	return boxes;
}

cv::Mat Salient::adaptBinarize(cv::Mat& im) {
	assert(im.channels() == 1);
	const int blockSize = 13;
	const int threshold = 11;
	const int sz = blockSize*blockSize;
	int halfSize = blockSize / 2;
	cv::Rect roi(halfSize, halfSize, im.cols, im.rows);
	cv::copyMakeBorder(im, im, halfSize, halfSize, halfSize, halfSize, cv::BORDER_CONSTANT, 0);

	cv::Mat iimage, biMap;
	cv::integral(im, iimage, CV_32S);
	biMap.create(im.size(), CV_8UC1);
	for (int j = halfSize; j < im.rows - halfSize - 1; ++j) {
		uchar* data = biMap.ptr(j);
		uchar* im_data = im.ptr(j);
		int* idata1 = iimage.ptr<int>(j - halfSize);
		int* idata2 = iimage.ptr<int>(j + halfSize + 1);
		for (int i = halfSize; i < im.cols - halfSize - 1; ++i) {
			int sum = (idata2[i + halfSize + 1] - idata2[i - halfSize] - idata1[i + halfSize + 1] + idata1[i - halfSize]);
			sum /= sz;
			if (im_data[i] < sum-threshold)
				data[i] = 0;
			else
				data[i] = 255;
		}
	}
	cv::Mat biMapNot;
	cv::bitwise_not(biMap(roi), biMapNot);
	return biMapNot;
	//return biMap(roi).clone();
}