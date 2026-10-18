#include "fc.h"
#include "fixed_point.h"

int fc_run(const FCLayer* layer,
           const uint8_t* input,
           uint8_t* output,
           TensorArena* arena) {
    
    const int in_features = layer->in_features;
    const int out_features = layer->out_features;
    
    // Per-channel or per-tensor quantization
    const int per_channel = (layer->weight_scale != NULL && layer->weight_scale[1] != 0.0f);
    
    for (int of = 0; of < out_features; of++) {
        int32_t acc = layer->bias[of];
        
        const float w_scale = per_channel ? layer->weight_scale[of] : layer->weight_scale[0];
        const int32_t w_zp = per_channel ? layer->weight_zp[of] : layer->weight_zp[0];
        
        const int32_t mult = layer->output_mult[of];
        const int shift = layer->output_shift[of];
        const int32_t out_zp = layer->output_zp;
        
        const int8_t* w_ptr = &layer->weight[of * in_features];
        
        for (int i = 0; i < in_features; i++) {
            uint8_t x_q = input[i];
            int8_t w_q = w_ptr[i];
            
            acc += (int32_t)w_q * (int32_t)x_q;
        }
        
        output[of] = requantize_u8(acc, mult, shift, out_zp);
    }
    
    return 0;
}