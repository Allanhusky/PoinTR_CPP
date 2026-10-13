#include "pointr/pointr_trt.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("Usage: pointr_benchmark ENGINE FILE_LIST OUTPUT.csv normalized|meters");
 bool normalized=std::string(argv[4])=="normalized";
 std::ifstream list(argv[2]);if(!list)throw std::runtime_error("Cannot read file list");
 std::vector<std::vector<pointr::Point>> clouds;std::string path;
 while(std::getline(list,path))if(!path.empty())clouds.push_back(pointr::read_cloud(path));
 if(clouds.empty())throw std::runtime_error("Empty input list");
 auto begin=std::chrono::steady_clock::now();pointr::PoinTrTRT model(argv[1]);
 double load=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
 for(int i=0;i<50;++i)model.infer(clouds[i%clouds.size()],normalized);
 std::ofstream out(argv[3]);if(!out)throw std::runtime_error("Cannot write CSV");
 out<<"sample_index,points,gpu_pipeline_ms,infer_ms\n"<<std::setprecision(10);
 for(size_t i=0;i<clouds.size();++i){auto start=std::chrono::steady_clock::now();auto result=model.infer(clouds[i],normalized);
 double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 if(result.points.size()!=8192)throw std::runtime_error("Unexpected output size");
 out<<i<<','<<clouds[i].size()<<','<<result.gpu_pipeline_ms<<','<<ms<<'\n';}
 std::cout<<"{\"samples\":"<<clouds.size()<<",\"warmup\":50,\"engine_load_ms\":"<<load<<"}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
