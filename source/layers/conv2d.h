#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>
#include "tensor_arena.h"

// Conv2D layer parameters (populated from model.h)
typedef struct {
    // Weights and bias (in ROM)
    const int8_t* weight;           // [out_ch, in_ch, kH, kW]
    const int32_t* bias;            // [out_ch]
    
    // Quantization parameters
    const float* weight_scale;      // [out_ch] per-channel
    const int32_t* weight_zp;       // [out_ch] (usually 0)
    
    float input_scale;              // scalar
    int32_t input_zp;               // scalar
    
    float output_scale;             // scalar
    int32_t output_zp;              // scalar
    
    const int32_t* output_mult;     // [out_ch] Q31 multiplier
    const int32_t* output_shift;    // [out_ch] usually 31
    
    // Shape
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride;
    int padding;
} Conv2DLayer;

// Run Conv2D: input [H, W, C_in] -> output [H_out, W_out, C_out]
// Input: UINT8 (NHWC layout)
// Output: UINT8 (NHWC layout)
// Returns 0 on success, -1 on arena OOM
int conv2d_run(const Conv2DLayer* layer, 
               const uint8_t* input, int H, int W,
               uint8_t* output, TensorArena* arena);

#endif // CONV2D_H