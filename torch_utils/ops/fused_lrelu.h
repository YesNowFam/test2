struct kernel_parameters {
    const void* input;
    const void* bias;
    const void* output;
    void* result;
    int input_numel;
    int input_stride;
    int bias_numel;
    bool activate;
    float alpha;
    float clamp;
};

template <class scalar_t> void* get_kernel();