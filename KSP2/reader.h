#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::vector<std::vector<int>> loadGtFromFile(const char* file);

std::vector<std::string> loadPathFromFile(const char* file);