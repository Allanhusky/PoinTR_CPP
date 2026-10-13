#include "pointr/point_cloud.hpp"
#include <filesystem>
#include <iostream>
#include <limits>
#include <cmath>
#include <stdexcept>
static void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(){
 try{
  std::vector<pointr::Point> p={{0,0,1},{.1f,0,1},{0,.1f,1},{.1f,.1f,1}};
  pointr::Normalization n;
  auto q=pointr::sample_and_normalize(p,n);require(q.size()==2048,"Sparse point count");
  pointr::denormalize(q,n);
  for(auto x:q){bool found=false;for(auto y:p)found|=std::abs(x.x-y.x)<1e-6&&std::abs(x.y-y.y)<1e-6&&std::abs(x.z-y.z)<1e-6;require(found,"Sparse inputs must repeat actual points");}
  auto dir=std::filesystem::temp_directory_path()/"pointr_cpp_cloud_test";std::filesystem::create_directories(dir);
  for(const auto& ext:{".npy",".ply",".xyz"}){
   auto f=(dir/((std::string)"cloud"+ext)).string();pointr::write_cloud(f,p);auto r=pointr::read_cloud(f);require(r.size()==p.size(),"Roundtrip size");
   for(size_t i=0;i<p.size();++i)require(std::abs(r[i].x-p[i].x)<1e-7&&std::abs(r[i].z-p[i].z)<1e-7,"Roundtrip values");
  }
  bool rejected=false;try{pointr::sample_and_normalize({{0,0,0}},n);}catch(...){rejected=true;}require(rejected,"Degenerate cloud rejection");
  rejected=false;try{pointr::sample_and_normalize(p,n,true);}catch(...){rejected=true;}require(rejected,"Normalized shape validation");
  rejected=false;try{pointr::sample_and_normalize({{std::numeric_limits<float>::infinity(),0,1}},n);}catch(...){rejected=true;}require(rejected,"Non-finite rejection");
  std::filesystem::remove_all(dir);std::cout<<"Cloud IO, sparse resampling, normalization and invalid input checks passed.\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
