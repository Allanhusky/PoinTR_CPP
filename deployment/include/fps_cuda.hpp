#pragma once
#include <cuda_runtime_api.h>
int fps_cuda(int b,int n,int m,const float* points,float* workspace,int* out,cudaStream_t stream);
