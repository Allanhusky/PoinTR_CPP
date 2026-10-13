#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include "fps_cuda.hpp"
#include <cstring>
#include <string>
#include <new>

namespace pointr {
class FPSPlugin final : public nvinfer1::IPluginV2DynamicExt {
    int count_;
    std::string ns_ = "pointr";
public:
    explicit FPSPlugin(int count):count_(count) {}
    const char* getPluginType() const noexcept override { return "FarthestPointSampling"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return sizeof(count_); }
    void serialize(void* buffer) const noexcept override { std::memcpy(buffer,&count_,sizeof(count_)); }
    void destroy() noexcept override { delete this; }
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* p=new(std::nothrow) FPSPlugin(count_); if(p) p->ns_=ns_; return p;
    }
    void setPluginNamespace(const char* ns) noexcept override { ns_=ns ? ns : ""; }
    const char* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::DataType getOutputDataType(int, const nvinfer1::DataType*, int) const noexcept override {
        return nvinfer1::DataType::kINT32;
    }
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs* inputs,int,nvinfer1::IExprBuilder& b) noexcept override {
        nvinfer1::DimsExprs out;out.nbDims=2;out.d[0]=inputs[0].d[0];out.d[1]=b.constant(count_);return out;
    }
    bool supportsFormatCombination(int pos,const nvinfer1::PluginTensorDesc* io,int nInputs,int nOutputs) noexcept override {
        if(nInputs!=1 || nOutputs!=1 || pos<0 || pos>1) return false;
        return io[pos].format==nvinfer1::TensorFormat::kLINEAR &&
            io[pos].type==(pos==0?nvinfer1::DataType::kFLOAT:nvinfer1::DataType::kINT32);
    }
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override {}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* in,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override {
        return static_cast<size_t>(in[0].dims.d[0])*in[0].dims.d[1]*sizeof(float);
    }
    int enqueue(const nvinfer1::PluginTensorDesc* in,const nvinfer1::PluginTensorDesc*,
                const void* const* inputs,void* const* outputs,void* workspace,cudaStream_t stream) noexcept override {
        auto d=in[0].dims;
        if(d.nbDims!=3 || d.d[2]!=3 || d.d[0]<1 || count_<1 || count_>d.d[1]) return 1;
        return fps_cuda(d.d[0],d.d[1],count_,static_cast<const float*>(inputs[0]),
                        static_cast<float*>(workspace),static_cast<int*>(outputs[0]),stream);
    }
};

class FPSCreator final : public nvinfer1::IPluginCreator {
    nvinfer1::PluginField field_{"count",nullptr,nvinfer1::PluginFieldType::kINT32,1};
    nvinfer1::PluginFieldCollection fields_{1,&field_};
    std::string ns_="pointr";
public:
    const char* getPluginName() const noexcept override { return "FarthestPointSampling"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(const char*,const nvinfer1::PluginFieldCollection* fields) noexcept override {
        int count=0;
        for(int i=0;fields && i<fields->nbFields;++i) {
            const auto& f=fields->fields[i];
            if(std::strcmp(f.name,"count")==0 && f.data && f.type==nvinfer1::PluginFieldType::kINT32 && f.length==1)
                count=*static_cast<const int*>(f.data);
        }
        if(count<1) return nullptr;
        auto* p=new(std::nothrow) FPSPlugin(count);if(p) p->setPluginNamespace(ns_.c_str());return p;
    }
    nvinfer1::IPluginV2* deserializePlugin(const char*,const void* data,size_t size) noexcept override {
        if(!data || size!=sizeof(int)) return nullptr;
        int count;std::memcpy(&count,data,sizeof(count));if(count<1) return nullptr;
        auto* p=new(std::nothrow) FPSPlugin(count);if(p)p->setPluginNamespace(ns_.c_str());return p;
    }
    void setPluginNamespace(const char* ns) noexcept override { ns_=ns?ns:""; }
    const char* getPluginNamespace() const noexcept override { return ns_.c_str(); }
};
}
namespace { pointr::FPSCreator& fps_creator() { static pointr::FPSCreator creator; return creator; } }
extern "C" bool register_pointr_plugins() {
    static bool registered=getPluginRegistry()->registerCreator(fps_creator(),"pointr");
    return registered;
}
#if NV_TENSORRT_MAJOR >= 10
extern "C" nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& count) {
    static nvinfer1::IPluginCreatorInterface* creators[]={&fps_creator()};count=1;return creators;
}
extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder*) {}
#else
namespace { const bool automatic_registration=register_pointr_plugins(); }
#endif
