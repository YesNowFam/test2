#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "fused_lrelu.h"

static torch::Tensor fused_lrelu(
    torch::Tensor x, 
    torch::Tensor b,
    torch::Tensor y,
    bool activate,
    float alpha, 
    float clamp
){
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    torch::Tensor r = torch::empty_like(x);
    bool backward = (y.numel() != 0);
    kernel_parameters params;

    params.input = x.data_ptr();
    params.bias = !(backward) ? b.data_ptr() : NULL;
    params.output = (backward) ? y.data_ptr() : NULL;
    params.result = r.data_ptr();
    params.input_numel = (int)x.numel();
    params.input_stride = (int)x.stride(1);
    params.bias_numel = !(backward) ? (int)b.numel() : NULL;
    params.activate = activate;
    params.alpha = alpha;
    params.clamp = clamp;

    void* kernel;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_lrelu_cuda", [&]
    {
        kernel = get_kernel<scalar_t>();
    });

    const int block_size = 128;
    const int grid_size = (params.input_numel - 1) / (4 * block_size) + 1;
    void* args[] = {&params};
    AT_CUDA_CHECK(cudaLaunchKernel(kernel, grid_size, block_size, args, 0, at::cuda::getCurrentCUDAStream()));
    return r;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_lrelu", &fused_lrelu);
}
