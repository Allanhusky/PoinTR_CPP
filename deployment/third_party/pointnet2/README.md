The FPS CUDA kernel in src/fps_cuda.cu is adapted from Pointnet2_PyTorch sampling_gpu.cu.
Original source: https://github.com/erikwijmans/Pointnet2_PyTorch
The original sampling order, origin-distance filter and reduction tie handling are retained.
See UNLICENSE. PyTorch/ATen dependencies were removed; TensorRT supplies the stream and workspace.
