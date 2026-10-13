#pragma once
#include <NvInfer.h>
#include <iostream>
namespace pointr {
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity,const char* msg) noexcept override {
        if(severity<=Severity::kWARNING) std::cerr << "[TensorRT] " << msg << '\n';
    }
};
}
