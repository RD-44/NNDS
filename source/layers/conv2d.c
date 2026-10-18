#include "conv2d.h"
#include "fixed_point.h"

int conv2d_run(const Conv2DLayer* layer, 
               const uint8_t* input, int H, int W,
               uint8_t* output, TensorArena* arena) {
    
    const int C_in = layer->in_channels;
    const int C_out = layer->out_channels;
    const int KH = layer->kernel_h;
    const int KW = layer->kernel_w;
    const int stride = layer->stride;
    const int pad = layer->padding;
    
    const int H_out = (H + 2 * pad - KH) / stride + 1;
    const int W_out = (W + 2 * pad - KW) / stride + 1;
    
    // For each output channel
    for (int oc = 0; oc < C_out; oc++) {
        const int32_t bias = layer->bias[oc];
        const int32_t mult = layer->output_mult[oc];
        const int shift = layer->output_shift[oc];
        const int32_t out_zp = layer->output_zp;
        
        const float w_scale = layer->weight_scale[oc];
        const int32_t w_zp = layer->weight_zp ? layer->weight_zp[oc] : 0;
        
        // Weight pointer for this output channel: [C_in, KH, KW]
        const int8_t* w_ptr = &layer->weight[oc * C_in * KH * KW];
        
        // For each output position
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                int32_t acc = bias;
                
                const int h_start = oh * stride - pad;
                const int w_start = ow * stride - pad;
                
                // Accumulate over kernel
                for (int kh = 0; kh < KH; kh++) {
                    const int ih = h_start + kh;
                    if (ih < 0 || ih >= H) continue;
                    
                    for (int kw = 0; kw < KW; kw++) {
                        const int iw = w_start + kw;
                        if (iw < 0 || iw >= W) continue;
                        
                        for (int ic = 0; ic < C_in; ic++) {
                            const int input_idx = (ih * W + iw) * C_in + ic;
                            const int weight_idx = (ic * KH + kh) * KW + kw;
                            
                            uint8_t x_q = input[input_idx];
                            int8_t w_q = w_ptr[weight_idx];
                            
                            // INT8 * UINT8 -> INT32
                            acc += (int32_t)w_q * (int32_t)x_q;
                        }
                    }
                }
                
                // Requantize to UINT8
                const int out_idx = (oh * W_out + ow) * C_out + oc;
                output[out_idx] = requantize_u8(acc, mult, shift, out_zp);
            }
        }
    }
    
    return 0;
}