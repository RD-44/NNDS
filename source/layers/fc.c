#include "fc.h"
#include "fixed_point.h"

int fc_run(const FCLayer* layer,
           const uint8_t* input,
           uint8_t* output,
           TensorArena* arena) {
    (void)arena;
    
    const int in_features = layer->in_features;
    const int out_features = layer->out_features;
    const int32_t in_zp = layer->input_zp;
    
    for (int of = 0; of < out_features; of++) {
        int32_t acc = layer->bias[of];
        
        const int32_t mult = layer->output_mult[of];
        const int shift = layer->output_shift[of];
        const int32_t out_zp = layer->output_zp;
        const int32_t w_zp = layer->weight_zp[of];
        
        const int8_t* w_ptr = &layer->weight[of * in_features];
        
        for (int i = 0; i < in_features; i++) {
            // Quantized values with zero points corrected
            const int32_t x_c = (int32_t)input[i] - in_zp;
            const int32_t w_c = (int32_t)w_ptr[i] - w_zp;
            
            acc += x_c * w_c;
        }
        
        output[of] = requantize_u8(acc, mult, shift, out_zp);
    }
    
    return 0;
}