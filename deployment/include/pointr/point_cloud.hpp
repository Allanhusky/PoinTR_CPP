#pragma once
#include <string>
#include <vector>
#include <cstdint>
namespace pointr {
struct Point { float x,y,z; };
struct Normalization { double x=0,y=0,z=0,radius=1; };
std::vector<Point> read_cloud(const std::string& path);
void write_cloud(const std::string& path,const std::vector<Point>& points);
std::vector<Point> sample_and_normalize(const std::vector<Point>& points,Normalization& norm,bool normalized=false,uint32_t seed=42);
void denormalize(std::vector<Point>& points,const Normalization& norm);
}
