#pragma once
#include <iostream>
#include <vector>
#include <queue>
#include "detection.h"
using namespace std;

using P = pair<float, int>;
struct edge {
	int to;
	float cost;
	float flow;
};

template <typename TNODE>
class Graph
{
public:
	Graph();
	~Graph();

public:
	int globalOffset;
	int totalSize;
	vector<int> localSizes;
	vector<vector<edge>> G;
	TNODE hyperSource;
	TNODE hyperTerminal;
	TNODE** nodes;
	int* next;
	float * dist;
	int* prev;
	char* searched;
	const int K = 10;
private:
	float calcFeatDistance(float* feat1, float* feat2, int len);

public:
	void buildGraph(TNODE** _nodes, vector<int>& sizes, int start, int radius);
	void dijkstra(int from, int to);
	void k_shortest();
	bool shortestPath();
	void deletePath(int from, int to);
	void localToGlobal(int* global_next);
	void validate();
};

template <typename TNODE>
Graph<TNODE>::Graph() {
	nodes = nullptr;
	dist = nullptr;
	prev = nullptr;
	next = nullptr;
	searched = nullptr;
	globalOffset = 0;
	totalSize = 0;
}

template <typename TNODE>
Graph<TNODE>::~Graph() {
	if (nodes)
		delete nodes;
	if (dist)
		delete dist;
	if (prev)
		delete prev;
	if (searched)
		delete searched;
}

template <typename TNODE>
void Graph<TNODE>::buildGraph(TNODE** _nodes, vector<int>& sizes, int start, int radius) {
	for (auto i = 0; i < start; ++i) {
		globalOffset += sizes[i];
	}
	for (auto i = start; i < start + radius; ++i) {
		localSizes.push_back(sizes[i]);
		totalSize += sizes[i];
	}
	totalSize += 2;
	cout << "Total nodes: " << totalSize << endl;
	nodes = new TNODE*[totalSize];
	for (auto i = 1; i < totalSize - 1; ++i) {
		//trueId是全局trueId
		//在求局部最短路径统一不用trueId，而直接使用结点索引
		nodes[i] = _nodes[globalOffset+i - 1];
	}
	G.resize(totalSize);
	dist = new float[totalSize];
	prev = new int[totalSize];
	next = new int[totalSize];
	searched = new char[totalSize];
	for (auto i = 0; i < totalSize; ++i) {
		dist[i] = 1e5;
	}
	int offset = 1;
	for (auto i = 0; i < localSizes.size(); ++i) {
		int nDetCurFrame = localSizes[i];
		int nDetNextFrame = i < localSizes.size() - 1 ? localSizes[i + 1] : 0;
		float sum = 0.0;
		for (auto j = 0; j < nDetCurFrame; ++j) {
			int id = j + offset;
			for (auto k = 0; k < nDetNextFrame; ++k) {
				edge e;
				e.to = k+offset+nDetCurFrame;
				e.cost= -log(nodes[id]->confidence / (1 - nodes[id]->confidence));
				e.flow = calcFeatDistance(nodes[id]->feature, nodes[e.to]->feature, nodes[id]->featLen);
				sum += e.flow;
				G[id].push_back(e);
			}
			for (auto k = 0; k < G[id].size(); ++k) {
				G[id][k].flow /= sum;
			}
			edge se;	//from hyperSource
			edge te;	//to hyperTerminal
			se.to = id;
			se.cost = 0;
			se.flow = 1;
			te.to = totalSize - 1;
			te.cost = 0;
			te.flow = 1;
			G[0].push_back(se);
			G[id].push_back(te);
			searched[id] = 0;
		}
		offset += nDetCurFrame;
	}
	nodes[0] = &hyperSource;
	nodes[totalSize - 1] = &hyperTerminal;
	nodes[0]->trueId = 0;
	nodes[totalSize - 1]->trueId = totalSize - 1;
	nodes[0]->confidence = 1.;
	nodes[totalSize - 1]->confidence = 1.;
	if(totalSize==2)
		G[0].push_back(edge{ totalSize - 1, 0,1 });
	hyperSource.trueId = -1;
	hyperTerminal.trueId = -1;
	searched[0] = 0;
	searched[totalSize - 1] = 0;
}

template <typename TNODE>
void Graph<TNODE>::dijkstra(int from, int to) {
	priority_queue<P, vector<P>, greater<P> > q;
	q.push(P(0, from));
	for (auto i = 0; i < totalSize; ++i)
		dist[i] = 1e5;
	dist[from] = 0;
	dist[to] = 1e5;
	while (!q.empty()) {
		P p = q.top();
		q.pop();
		int v = p.second;
		if (dist[v]<p.first){
			//cout << "dist[" << v << "]: " << dist[v] << " ,p.first: " << p.first << endl;
			continue;
		}
		for (int i = 0; i<G[v].size(); i++) {
			edge& e = G[v][i];
			if (searched[e.to]) {
				continue;
			}
			if (dist[e.to]>dist[v] + e.cost*e.flow) {
				dist[e.to] = dist[v] + e.cost*e.flow;
				prev[e.to] = v;
				//cout << "Node v: " << v << " ,to " << e.to << " dist changed to: " << dist[e.to] << endl;
				if (e.to != to)
					q.push(P(dist[e.to], e.to));
			}
		}
	}
	for (auto v = to; v != from; v = prev[v]) {
		cout <<"["<<v<<","<<prev[v] << "],";
		if (v != totalSize - 1) {
			searched[v] = 1;
		}
		next[prev[v]] = v;
	}
	cout << endl;
}

template <typename TNODE>
void Graph<TNODE>::k_shortest() {
	if (totalSize == 0)
		return;
	int k = 0;
	for (auto i = 0; i < K; ++i) {
		if (!shortestPath()) {
			++k;
			break;
		}
		for (auto i = 1; i < totalSize - 1; ++i)
			cout << (int)searched[i] << ",";
		++k;
		cout << endl;
	}
	cout << "Total " << k << " shortest path found. Total nodes: " << totalSize << endl;
	//for (auto i = 1; i < totalSize - 1; ++i)
	//	searched[i]=1;
}

template <typename TNODE>
bool Graph<TNODE>::shortestPath() {
	bool hasPath = false;
	dijkstra(0, totalSize - 1);
	for (auto i = 1; i<totalSize - 1; ++i) {
		if (!searched[i]) {
			hasPath = true;
			break;
		}
	}
	return hasPath;
}

template <typename TNODE>
void Graph<TNODE>::deletePath(int from, int to) {
	int v = to;
	while (v != from) {
		searched[v] = 1;
		v = prev[v];
	}
}

template<typename TNODE>
inline void Graph<TNODE>::localToGlobal(int * global_next)
{
	for (auto i = 1; i < totalSize - 1; ++i) {
		//global_next[nodes[i]->trueId] = next[i]-1 + globalOffset;
		global_next[nodes[i]->trueId] = nodes[next[i]]->trueId;
	}
}

template<typename TNODE>
inline void Graph<TNODE>::validate()
{
	cout << "Total nodes: " << totalSize - 2 << endl;
	for (auto i = 1; i < totalSize - 1; ++i) {
		cout << "globalOffset+index: " << globalOffset + i - 1 << endl;
		cout << "trueId: " << nodes[i]->trueId << endl;
	}
}

template <typename TNODE>
float Graph<TNODE>::calcFeatDistance(float* feat1, float* feat2, int len) {
	float dist = 0.0;
	for (auto i = 0; i < len; ++i) {
		dist += (feat1[i] - feat2[i])*(feat1[i] - feat2[i]);
	}
	return 1.0 / sqrt(dist);
}