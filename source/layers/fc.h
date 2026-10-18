#ifndef FC_H
#define FC_H

#include <stdint.h>
#include "tensor_arena.h"

typedef struct {
    const int8_t* weight;           // [out_features, in_features]
    const int32_t* bias;            // [out_features]
    
    const float* weight_scale;      // [1] or [out_features]
    const int32_t* weight_zp;       // [1] or [out_features]
    
    float input_scale;
    int32_t input_zp;
    
    float output_scale;
    int32_t output_zp;
    
    const int32_t* output_mult;     // [out_features]
    const int32_t* output_shift;    // [out_features]
    
    int in_features;
    int out_features;
} FCLayer;

int fc_run(const FCLayer* layer,
           const uint8_t* input,
           uint8_t* output,
           TensorArena* arena);

#endif // FC_H