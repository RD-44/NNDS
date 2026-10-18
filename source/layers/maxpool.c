#include "maxpool.h"
#include "fixed_point.h"

int maxpool_run(const MaxPoolLayer* layer,
                const uint8_t* input, int H, int W, int C,
                uint8_t* output) {
    
    const int K = layer->kernel_size;
    const int stride = layer->stride;
    const int pad = layer->padding;
    
    const int H_out = (H + 2 * pad - K) / stride + 1;
    const int W_out = (W + 2 * pad - K) / stride + 1;
    
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                uint8_t max_val = 0;
                
                const int h_start = oh * stride - pad;
                const int w_start = ow * stride - pad;
                
                for (int kh = 0; kh < K; kh++) {
                    const int ih = h_start + kh;
                    if (ih < 0 || ih >= H) continue;
                    
                    for (int kw = 0; kw < K; kw++) {
                        const int iw = w_start + kw;
                        if (iw < 0 || iw >= W) continue;
                        
                        const int idx = (ih * W + iw) * C + c;
                        uint8_t val = input[idx];
                        if (val > max_val) max_val = val;
                    }
                }
                
                const int out_idx = (oh * W_out + ow) * C + c;
                output[out_idx] = max_val;
            }
        }
    }
    
    return 0;
}