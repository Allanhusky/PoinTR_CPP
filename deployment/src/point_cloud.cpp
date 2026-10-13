#include "pointr/point_cloud.hpp"
#include <fstream>
#include <sstream>
#include <regex>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <iomanip>
namespace pointr {
static bool finite(const Point& p) { return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z); }
std::vector<Point> read_cloud(const std::string& path) {
 std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot open cloud: "+path);
 std::vector<Point> points;
 auto ext=std::filesystem::path(path).extension().string();
 if(ext==".npy") {
  char magic[6];f.read(magic,6);if(std::memcmp(magic,"\x93NUMPY",6))throw std::runtime_error("Invalid NPY magic");
  unsigned char ver[2];f.read(reinterpret_cast<char*>(ver),2);
  uint32_t length=0;
  if(ver[0]==1){uint16_t n=0;f.read(reinterpret_cast<char*>(&n),2);length=n;}
  else if(ver[0]==2||ver[0]==3)f.read(reinterpret_cast<char*>(&length),4);
  else throw std::runtime_error("Unsupported NPY version");
  if(length>1024*1024)throw std::runtime_error("Invalid NPY header length");
  std::string header(length,' ');f.read(header.data(),length);
  std::smatch match;
  if(!std::regex_search(header,match,std::regex("'descr'\\s*:\\s*'([<|=][f][48])'")))throw std::runtime_error("NPY must be little-endian float32 or float64");
  const int bytes=match[1].str().back()-'0';
  if(header.find("True")!=std::string::npos)throw std::runtime_error("NPY must be C-contiguous");
  if(!std::regex_search(header,match,std::regex("'shape'\\s*:\\s*\\(\\s*([0-9]+)\\s*,\\s*3\\s*,?\\s*\\)")))throw std::runtime_error("NPY must have shape (N,3)");
  size_t n=std::stoull(match[1]);if(n==0||n>10000000)throw std::runtime_error("Invalid point count");
  points.resize(n);
  for(auto& p:points){double v[3];for(int k=0;k<3;++k){if(bytes==4){float x;f.read(reinterpret_cast<char*>(&x),4);v[k]=x;}else f.read(reinterpret_cast<char*>(&v[k]),8);}p={float(v[0]),float(v[1]),float(v[2])};}
  if(!f)throw std::runtime_error("Truncated NPY payload");
 } else if(ext==".ply") {
  std::string line;if(!std::getline(f,line)||line!="ply")throw std::runtime_error("Invalid PLY");
  size_t n=0;bool ascii=false,vertex=false,end=false;std::vector<std::string> properties;
  while(std::getline(f,line)) {
   if(line=="format ascii 1.0")ascii=true;
   std::istringstream ss(line);std::string key;ss>>key;
   if(key=="element"){std::string kind;size_t count;ss>>kind>>count;vertex=kind=="vertex";if(vertex)n=count;}
   if(key=="property"&&vertex){std::string type,name;ss>>type>>name;if(type=="list")throw std::runtime_error("List vertex properties unsupported");properties.push_back(name);}
   if(line=="end_header"){end=true;break;}
  }
  if(!ascii||!end||!n||n>10000000)throw std::runtime_error("Use an ASCII PLY with vertices");
  int ids[3];const char* names[]={"x","y","z"};
  for(int k=0;k<3;++k){auto it=std::find(properties.begin(),properties.end(),names[k]);if(it==properties.end())throw std::runtime_error("PLY missing XYZ");ids[k]=it-properties.begin();}
  for(size_t i=0;i<n;++i){if(!std::getline(f,line))throw std::runtime_error("Truncated PLY");std::istringstream ss(line);std::vector<float> row(properties.size());for(auto& x:row)if(!(ss>>x))throw std::runtime_error("Invalid PLY row");points.push_back({row[ids[0]],row[ids[1]],row[ids[2]]});}
 } else {
  std::string line;while(std::getline(f,line)){if(line.empty()||line[0]=='#')continue;std::istringstream ss(line);Point p;if(!(ss>>p.x>>p.y>>p.z))throw std::runtime_error("Invalid XYZ row");points.push_back(p);}
 }
 if(points.empty())throw std::runtime_error("Empty point cloud");
 for(const auto& p:points)if(!finite(p))throw std::runtime_error("Point cloud contains NaN/Inf");
 return points;
}
void write_cloud(const std::string& path,const std::vector<Point>& points) {
 auto parent=std::filesystem::path(path).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
 std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot write cloud: "+path);
 auto ext=std::filesystem::path(path).extension().string();
 if(ext==".npy") {
  std::string h="{'descr': '<f4', 'fortran_order': False, 'shape': ("+std::to_string(points.size())+", 3), }";
  h.append((64-(10+h.size()+1)%64)%64,' ');h+='\n';uint16_t length=h.size();
  f.write("\x93NUMPY",6);const char version[]={1,0};f.write(version,2);f.write(reinterpret_cast<const char*>(&length),2);f.write(h.data(),h.size());
  for(const auto& p:points){float xyz[]={p.x,p.y,p.z};f.write(reinterpret_cast<const char*>(xyz),sizeof(xyz));}
 } else {
  if(ext==".ply")f<<"ply\nformat ascii 1.0\nelement vertex "<<points.size()<<"\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
  f<<std::setprecision(9);for(const auto& p:points)f<<p.x<<' '<<p.y<<' '<<p.z<<'\n';
 }
 if(!f)throw std::runtime_error("Cloud write failed");
}
std::vector<Point> sample_and_normalize(const std::vector<Point>& input,Normalization& norm,bool normalized,uint32_t seed) {
 if(input.empty())throw std::runtime_error("Empty cloud");
 for(const auto& p:input)if(!finite(p))throw std::runtime_error("NaN/Inf input");
 if(normalized&&input.size()!=2048)throw std::runtime_error("Normalized input requires exactly 2048 points");
 std::vector<Point> points;points.reserve(2048);std::mt19937 rng(seed);
 if(input.size()==2048)points=input;
 else if(input.size()>2048){std::vector<size_t> ids(input.size());std::iota(ids.begin(),ids.end(),0);std::shuffle(ids.begin(),ids.end(),rng);for(size_t i=0;i<2048;++i)points.push_back(input[ids[i]]);}
 else{points=input;std::uniform_int_distribution<size_t> pick(0,input.size()-1);while(points.size()<2048)points.push_back(input[pick(rng)]);}
 norm=Normalization{};if(normalized)return points;
 for(const auto& p:points){norm.x+=p.x;norm.y+=p.y;norm.z+=p.z;}norm.x/=2048;norm.y/=2048;norm.z/=2048;
 norm.radius=0;for(const auto& p:points)norm.radius=std::max(norm.radius,std::sqrt(std::pow(p.x-norm.x,2)+std::pow(p.y-norm.y,2)+std::pow(p.z-norm.z,2)));
 if(norm.radius<1e-8||!std::isfinite(norm.radius))throw std::runtime_error("Degenerate point cloud");
 for(auto& p:points)p={float((p.x-norm.x)/norm.radius),float((p.y-norm.y)/norm.radius),float((p.z-norm.z)/norm.radius)};
 return points;
}
void denormalize(std::vector<Point>& points,const Normalization& norm) {
 for(auto& p:points)p={float(p.x*norm.radius+norm.x),float(p.y*norm.radius+norm.y),float(p.z*norm.radius+norm.z)};
}
}
