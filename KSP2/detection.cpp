#include "stdafx.h"
#include "detection.h"


Detection::Detection()
{
	nodes = nullptr;
	prev = nullptr;
	next = nullptr;
	ids = nullptr;
	totalSize = 0;
}


Detection::~Detection()
{
	for (auto i = 0; i < features.size(); ++i)
		delete features[i];
	if (nodes)
		delete nodes;
	if (prev)
		delete prev;
	if (next)
		delete next;
	if (ids)
		delete ids;
}

void Detection::detect(cv::Mat & image, vector<target>& frame_detections, int offset, int scale)
{
	cv::Mat saMap = salient.salientDetectFT(image);
	cv::Mat biMap = salient.adaptBinarize(saMap);
	vector<std::pair<cv::Rect, float>> bboxes = salient.findBoundingBoxes(biMap);

	for (auto i = 0; i < bboxes.size(); ++i) {
		target t;
		t.box = bboxes[i].first;
		//t.box.x = (int)t.box.x*scale;
		//t.box.y = (int)t.box.y*scale;
		//t.box.height = (int)t.box.height*scale;
		//t.box.width = (int)t.box.width*scale;
		t.frameId = i;
		t.trueId = i + offset;
		t.confidence = bboxes[i].second;
		t.featLen = featLen;
		float* feature = new float[featLen];
		extract(image(t.box), feature);
		t.feature = feature;
		features.push_back(feature);
		t.type = vehicle;
		frame_detections.push_back(t);
		cv::rectangle(image, t.box, cv::Scalar(0, 255, 255), 2);
	}
	cv::imshow("", image);
	cv::waitKey(1);
}

void Detection::extract(cv::Mat & roi, float * feat)
{
	ex.calcHist(roi, feat);
}


void Detection::runVideo(const char* filename) {
	file_paths = loadPathFromFile(filename);
	this->filename = string(filename);
	cv::Mat frame;
	int offset = 0;
	int scale = 2;
	//
	for (auto i = 0; i<file_paths.size(); ++i) {
		frame = cv::imread(file_paths[i]);
		cv::Mat scaled;
		cv::resize(frame, scaled, cv::Size(frame.cols / scale, frame.rows / scale));

		vector<target> frame_detections;
		detect(scaled, frame_detections, offset, scale);

		detections.push_back(frame_detections);
		frameSizes.push_back(frame_detections.size());

		offset += frame_detections.size();
		totalSize += frame_detections.size();
	}
	nodes = new target*[totalSize];
	next = new int[totalSize];
	prev = new int[totalSize];
	ids = new int[totalSize];
	int cnt = 0;
	for (auto i = 0; i < detections.size(); ++i) {
		for (auto j = 0; j < detections[i].size(); ++j) {
			nodes[cnt] = &(detections[i][j]);
			//cout << "Raw trueId: " << nodes[cnt]->trueId << endl;
			++cnt;
		}
	}
}

void Detection::parseDisjointPaths() {
	int id = 0;
	cout << "Total " << totalSize << " targets wrapped" << endl;
	for (auto i = 0; i < totalSize; ++i) {
		if (nodes[i]->searched) {
			cout << "Fucking searched?" << endl;
			continue;
		}
		int v = i;
		vector<int> traj;
		while (1) {
			ids[v] = id;
			traj.push_back(v);
			nodes[v]->searched = true;
			v = next[v];
			if (v == -1) {
				cout << "trajectory terminated" << endl;
				break;
			}
		}
		disjointPaths.push_back(traj);
		id = (id + 1) % MAX_T;
	}
	cout << "Total " << disjointPaths.size() << " disjoint trajectories" << endl;
}

void Detection::drawBoundingBoxes()
{
	int scale = 2;
	cv::Mat frame;
	int offset = 0;
	for (auto k = 0; k < file_paths.size(); ++k) {
		frame = cv::imread(file_paths[k]);
		cv::Mat scaled;
		cv::resize(frame, scaled, cv::Size(frame.cols / scale, frame.rows / scale), cv::INTER_LINEAR);

		for (int i = 0; i < frameSizes[k]; ++i) {
			int id = i + offset;
			cv::Rect& box = nodes[id]->box;
			cout << ids[id] << endl;
			cv::rectangle(scaled, box, colors[ids[id]], 2);
		}
		offset += frameSizes[k];

		cv::imshow("box", scaled);
		cv::waitKey(50);
	}
	
}

