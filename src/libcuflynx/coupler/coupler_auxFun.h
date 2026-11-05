#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>


#pragma once

class AuxCouplerFun {
public:

    std::vector<double> extractDoublesFromBytes(const char* byteBuffer, size_t bufferSize, size_t nBytes);
    void convertDoublesToBytes(const std::vector<double>& doubleVec, char* byteBuffer, size_t bufferSize);

    std::string serializeToJson(const std::map<std::string, std::map<std::string, int>>& myMap);
    std::map<std::string, std::map<std::string, int>> deserializeFromJson(const std::string& jsonStr);
    void saveJsonToFile(const std::string& filename, const std::string& jsonStr);
    std::string readJsonFromFile(const std::string& filename);

};
