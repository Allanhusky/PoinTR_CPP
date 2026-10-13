#include "pointr/pointr_trt.hpp"
#include "pointr/trt_logger.hpp"
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <stdexcept>
#include <cmath>
namespace pointr {
extern "C" bool register_pointr_plugins();
static void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
template<class T> static void release(T*& p){if(!p)return;
#if NV_TENSORRT_MAJOR >= 10
 delete p;
#else
 p->destroy();
#endif
 p=nullptr;}
struct PoinTrTRT::Impl {
 TrtLogger logger;
 nvinfer1::IRuntime* runtime=nullptr;
 nvinfer1::ICudaEngine* engine=nullptr;
 nvinfer1::IExecutionContext* context=nullptr;
 cudaStream_t stream=nullptr;
 cudaEvent_t begin=nullptr,end=nullptr;
 float* input=nullptr;float* output=nullptr;
 std::string input_name,output_name;
 ~Impl(){if(stream)cudaStreamSynchronize(stream);if(input)cudaFree(input);if(output)cudaFree(output);
  if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);
  release(context);release(engine);release(runtime);}
 void init(const std::string& path){
  if(!register_pointr_plugins())throw std::runtime_error("FPS plugin registration failed");
  std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Cannot open engine: "+path);
  auto size=f.tellg();if(size<=0)throw std::runtime_error("Empty engine");std::vector<char> bytes(static_cast<size_t>(size));f.seekg(0);f.read(bytes.data(),bytes.size());if(!f)throw std::runtime_error("Engine read failed");
  runtime=nvinfer1::createInferRuntime(logger);if(!runtime)throw std::runtime_error("createInferRuntime failed");
  engine=runtime->deserializeCudaEngine(bytes.data(),bytes.size());if(!engine)throw std::runtime_error("Engine deserialization failed; rebuild engine for this GPU/TRT version");
  context=engine->createExecutionContext();if(!context)throw std::runtime_error("Context creation failed");
  if(engine->getNbIOTensors()!=2)throw std::runtime_error("Expected exactly one input and one output");
  for(int i=0;i<engine->getNbIOTensors();++i){const char* n=engine->getIOTensorName(i);
   if(engine->getTensorDataType(n)!=nvinfer1::DataType::kFLOAT)throw std::runtime_error("Expected FP32 input/output bindings");
   if(engine->getTensorIOMode(n)==nvinfer1::TensorIOMode::kINPUT)input_name=n;else output_name=n;}
  if(input_name.empty()||output_name.empty())throw std::runtime_error("Invalid engine IO modes");
  auto in=engine->getTensorShape(input_name.c_str()),out=engine->getTensorShape(output_name.c_str());
  if(in.nbDims!=3||in.d[0]!=1||in.d[1]!=2048||in.d[2]!=3||out.nbDims!=3||out.d[0]!=1||out.d[1]!=8192||out.d[2]!=3)throw std::runtime_error("Require fixed input [1,2048,3] and output [1,8192,3]");
  check(cudaStreamCreate(&stream));check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
  check(cudaMalloc(reinterpret_cast<void**>(&input),2048*3*sizeof(float)));check(cudaMalloc(reinterpret_cast<void**>(&output),8192*3*sizeof(float)));
  if(!context->setTensorAddress(input_name.c_str(),input)||!context->setTensorAddress(output_name.c_str(),output))throw std::runtime_error("setTensorAddress failed");
 }
};
PoinTrTRT::PoinTrTRT(const std::string& path):impl_(new Impl){impl_->init(path);}
PoinTrTRT::~PoinTrTRT()=default;
Completion PoinTrTRT::infer(const std::vector<Point>& points,bool normalized,uint32_t seed){
 static_assert(sizeof(Point)==3*sizeof(float),"Point must contain packed XYZ");
 Completion result;auto input=sample_and_normalize(points,result.normalization,normalized,seed);result.points.resize(8192);
 auto& p=*impl_;
 check(cudaEventRecord(p.begin,p.stream));
 check(cudaMemcpyAsync(p.input,input.data(),input.size()*sizeof(Point),cudaMemcpyHostToDevice,p.stream));
 if(!p.context->enqueueV3(p.stream))throw std::runtime_error("TensorRT enqueueV3 failed");
 check(cudaMemcpyAsync(result.points.data(),p.output,result.points.size()*sizeof(Point),cudaMemcpyDeviceToHost,p.stream));
 check(cudaEventRecord(p.end,p.stream));check(cudaEventSynchronize(p.end));check(cudaEventElapsedTime(&result.gpu_pipeline_ms,p.begin,p.end));
 for(const auto& q:result.points)if(!std::isfinite(q.x)||!std::isfinite(q.y)||!std::isfinite(q.z))throw std::runtime_error("Non-finite model output");
 if(!normalized)denormalize(result.points,result.normalization);
 return result;
}
}
