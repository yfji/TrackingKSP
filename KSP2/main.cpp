// KSP2.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "graph.hpp"

void ksp() {
	Detection d;
	d.runVideo("I:/Experiment/dataset/pafiss_eval_dataset/sequence03_files.txt");
	int radius = 5;
	int offset = 0;
	for (auto i = 0; i < d.detections.size() - 1;) {
		Graph<target> graph;
		graph.buildGraph(d.nodes, d.frameSizes, i, radius);
		graph.k_shortest();
		graph.localToGlobal(d.next);
		i += (radius - 1);
		if (i + 2 * radius>d.detections.size()) {
			radius = d.detections.size() - i;
		}
	}
	cout << endl << "Generating trajectories..." << endl;
	d.parseDisjointPaths();
	d.drawBoundingBoxes();
}

int main()
{
	ksp();
    return 0;
}

