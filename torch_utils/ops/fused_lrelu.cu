#include <c10/util/Half.h>
#include "fused_lrelu.h"

template <class scalar_t>
__global__ void fused_lrelu_kernel(kernel_parameters params) {
    float gain = 1.4142135623730951;
    bool backward = (params.output != NULL);
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;

    for (int i = 0; i < 4 && idx < params.input_numel; i++, idx += blockDim.x)
    {
        float x = (float)((const scalar_t*)params.input)[idx];
        if (params.bias != NULL) { 
            float b = (float)((const scalar_t*)params.bias)[(idx / params.input_stride) % params.bias_numel];
            x = x + b;
        }

        float y = x;
        if (backward) {
            y = (float)((const scalar_t*)params.output)[idx];
        }

        if (params.activate) {
            if (y <= 0) x = x * params.alpha;
            x = x * gain;
        }

        y = (backward) ? y : x;

        if (params.clamp >= 0 && (y < -params.clamp || y > params.clamp)) {
            x = (backward) ? 0 : ((y < 0) ? -params.clamp : params.clamp);
        }

        ((scalar_t*)params.result)[idx] = (scalar_t)x;
    }
}

template <class scalar_t> void* get_kernel()
{
    return (void*)fused_lrelu_kernel<scalar_t>;
}

template void* get_kernel<double>();
template void* get_kernel<float>();
template void* get_kernel<c10::Half>();
