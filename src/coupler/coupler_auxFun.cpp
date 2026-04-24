#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>

#include "coupler_auxFun.h"


std::vector<double> AuxCouplerFun::extractDoublesFromBytes(const char* byteBuffer, size_t bufferSize, size_t nBytes)
{
    // Validate the number of bytes to convert
    if (nBytes > bufferSize) {
        throw std::invalid_argument("Number of bytes to convert exceeds buffer size.");
    }
    // Ensure the input number of bytes is a multiple of the size of a double
    if (nBytes % sizeof(double) != 0) {
        throw std::invalid_argument("Number of bytes is not a multiple of double size.");
    }
    size_t nn = nBytes/sizeof(double);
    // Convert bytes to doubles
    std::vector<double> result;
    for (size_t i = 0; i < nn; ++i) {
        double value;
        std::memcpy(&value, byteBuffer + i * sizeof(double), sizeof(double)); // std::memcpy is used to copy raw bytes into a double
        result.push_back(value);
    }
    return result;
}

void AuxCouplerFun::convertDoublesToBytes(const std::vector<double>& doubleVec, char* byteBuffer, size_t bufferSize)
{
    size_t nDoubles = doubleVec.size();
    size_t nBytes = nDoubles * sizeof(double);
    // Validate that the provided buffer is large enough
    if (bufferSize < nBytes) {
        throw std::invalid_argument("Buffer is too small to hold the doubles.");
    }
    // Copy each double into the buffer as raw bytes
    for (size_t i = 0; i < nDoubles; ++i) {
        std::memcpy(byteBuffer + i * sizeof(double), &doubleVec[i], sizeof(double));
    }
}


// Serialize a nested map to a JSON-like string
std::string AuxCouplerFun::serializeToJson(const std::map<std::string, std::map<std::string, int>>& myMap)
{
    std::ostringstream json;
    json << "{\n";
    bool firstCategory = true;
    
    for (const auto& category : myMap) {
        if (!firstCategory) json << ",\n";
        json << "  \"" << category.first << "\": {\n";
        
        bool firstItem = true;
        for (const auto& pair : category.second) {
            if (!firstItem) json << ",\n";
            json << "    \"" << pair.first << "\": " << pair.second;
            firstItem = false;
        }
        
        json << "\n  }";
        firstCategory = false;
    }
    
    json << "\n}";
    return json.str();
}

// Deserialize a JSON-like string to a nested map
std::map<std::string, std::map<std::string, int>> AuxCouplerFun::deserializeFromJson(const std::string& jsonStr)
{
    std::map<std::string, std::map<std::string, int>> myMap;
    std::string category;
    std::string key;
    int value;
    
    std::istringstream json(jsonStr); // this allows reading from the JSON string like a file
    char ch; // temporary variable used to read individual characters
    
    while (json >> ch) { // iterate and read the JSON string character by character
        if (ch == '"') {  // Start reading category name
            std::getline(json, category, '"');
            json >> ch; // Skip ':'
            json >> ch; // Skip '{'

            std::map<std::string, int> innerMap;
            while (json >> ch && ch != '}') {  // Read key-value pairs inside category
                if (ch == '"') {
                    std::getline(json, key, '"');  // Read key name
                    json >> ch; // Skip ':'
                    json >> value;
                    innerMap[key] = value;
                }
                // Skip ',' but be careful not to accidentally consume '}' at the end
                if (json.peek() == ',') json.ignore();
            }
            myMap[category] = innerMap;
        }
    }

    return myMap;
}

// Save JSON string to a file
void AuxCouplerFun::saveJsonToFile(const std::string& filename, const std::string& jsonStr)
{
    std::ofstream file(filename);
    if (file.is_open()) {
        file << jsonStr;
        file.close();
    } else {
        std::cerr << "Error: Could not open JSON file " << filename << " for writing!" << std::endl;
    }
}

// Read JSON string from a file
std::string AuxCouplerFun::readJsonFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open JSON file " << filename << " for reading!" << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf(); // Read file contents into stringstream
    file.close();
    return buffer.str();
}
