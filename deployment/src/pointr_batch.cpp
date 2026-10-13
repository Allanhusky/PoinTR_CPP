#include "pointr/pointr_trt.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <stdexcept>
int main(int argc,char** argv){
 try{
  std::string engine,input,output;bool normalized=false;int warmup=3,repeat=10;
  for(int i=1;i<argc;++i){std::string a=argv[i];
   if(a=="--help"){std::cout<<"pointr_batch --engine MODEL.engine --input CLOUD.npy|ply|xyz --output COMPLETION.npy|ply [--normalized] [--warmup 3] [--repeat 10]\nInput: camera optical XYZ in meters. --normalized: exactly 2048 normalized points, normalized output.\n";return 0;}
   if(a=="--normalized"){normalized=true;continue;}
   if(i+1>=argc)throw std::runtime_error("Missing value for "+a);
   std::string v=argv[++i];if(a=="--engine")engine=v;else if(a=="--input")input=v;else if(a=="--output")output=v;else if(a=="--warmup")warmup=std::stoi(v);else if(a=="--repeat")repeat=std::stoi(v);else throw std::runtime_error("Unknown option: "+a);
  }
  if(engine.empty()||input.empty()||output.empty()||warmup<0||repeat<1)throw std::runtime_error("Invalid arguments; use --help");
  auto points=pointr::read_cloud(input);pointr::PoinTrTRT model(engine);
  for(int i=0;i<warmup;++i)model.infer(points,normalized);
  pointr::Completion result;double gpu=0;auto start=std::chrono::steady_clock::now();
  for(int i=0;i<repeat;++i){result=model.infer(points,normalized);gpu+=result.gpu_pipeline_ms;}
  double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeat;
  pointr::write_cloud(output,result.points);
  auto n=result.normalization;
  std::cout<<std::setprecision(9)<<"{\"raw_input_points\":"<<points.size()<<",\"output_points\":"<<result.points.size()<<",\"gpu_pipeline_ms\":"<<gpu/repeat
   <<",\"end_to_end_infer_ms\":"<<elapsed<<",\"centroid\":["<<n.x<<','<<n.y<<','<<n.z<<"],\"radius_m\":"<<n.radius<<",\"normalized\":"<<(normalized?"true":"false")<<"}\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
