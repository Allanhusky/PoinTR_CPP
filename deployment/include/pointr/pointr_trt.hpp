#pragma once
#include "pointr/point_cloud.hpp"
#include <memory>
#include <string>
namespace pointr {
struct Completion {
 std::vector<Point> points;
 Normalization normalization;
 float gpu_pipeline_ms=0;
};
// One instance owns one CUDA stream/context. Use separate instances for concurrent calls.
class PoinTrTRT {
public:
 explicit PoinTrTRT(const std::string& engine_path);
 ~PoinTrTRT();
 PoinTrTRT(const PoinTrTRT&)=delete;
 PoinTrTRT& operator=(const PoinTrTRT&)=delete;
 Completion infer(const std::vector<Point>& camera_points_m,bool input_normalized=false,uint32_t seed=42);
private:
 struct Impl;
 std::unique_ptr<Impl> impl_;
};
}
