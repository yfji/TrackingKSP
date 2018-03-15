#include "stdafx.h"
#include "reader.h"

std::vector<std::vector<int>> loadGtFromFile(const char* file) {
	std::ifstream in;
	std::vector<std::vector<int>> gts;
	in.open(file, std::ios::in);
	if (!in) {
		std::cerr << "File not exists" << std::endl;
		return gts;
	}
	while (!in.eof()) {
		std::vector<int> gt(4);
		std::string line;
		std::getline(in, line);
		if (line.length() == 0)
			break;
		std::stringstream ss(line);
		for (auto i = 0; i < 4; ++i) {
			ss >> gt[i];
		}
		gts.push_back(gt);
	}
	in.close();
	return gts;
}

std::vector<std::string> loadPathFromFile(const char* file) {
	std::ifstream in;
	std::vector<std::string> file_paths;
	in.open(file, std::ios::in);
	if (!in) {
		std::cerr << "File not exists" << std::endl;
		return file_paths;
	}
	while (!in.eof()) {
		std::string line;
		std::getline(in, line);
		if (line.length() == 0)
			break;
		file_paths.push_back(line);
	}
	in.close();
	return file_paths;
}