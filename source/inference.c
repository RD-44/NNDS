#include "inference.h"
#include "layers/conv2d.h"
#include "layers/maxpool.h"
#include "layers/fc.h"
#include "model.h"
#include "tensor_arena.h"
#include "fixed_point.h"

// Global arena
static TensorArena g_arena;
static size_t g_peak_arena_usage = 0;

// Layer declarations from model.h
extern const int8_t conv1_weight[];
extern const int32_t conv1_bias[];
extern const float conv1_weight_scale[];
extern const int32_t conv1_weight_zp[];
extern const int32_t conv1_output_mult[];
extern const int32_t conv1_output_shift[];
extern const float conv1_input_scale;
extern const int32_t conv1_input_zp;
extern const float conv1_output_scale;
extern const int32_t conv1_output_zp;

extern const int8_t conv2_weight[];
extern const int32_t conv2_bias[];
extern const float conv2_weight_scale[];
extern const int32_t conv2_weight_zp[];
extern const int32_t conv2_output_mult[];
extern const int32_t conv2_output_shift[];
extern const float conv2_input_scale;
extern const int32_t conv2_input_zp;
extern const float conv2_output_scale;
extern const int32_t conv2_output_zp;

extern const int8_t fc1_weight[];
extern const int32_t fc1_bias[];
extern const float fc1_weight_scale[];
extern const int32_t fc1_weight_zp[];
extern const int32_t fc1_output_mult[];
extern const int32_t fc1_output_shift[];
extern const float fc1_input_scale;
extern const int32_t fc1_input_zp;
extern const float fc1_output_scale;
extern const int32_t fc1_output_zp;

extern const int8_t fc2_weight[];
extern const int32_t fc2_bias[];
extern const float fc2_weight_scale[];
extern const int32_t fc2_weight_zp[];
extern const int32_t fc2_output_mult[];
extern const int32_t fc2_output_shift[];
extern const float fc2_input_scale;
extern const int32_t fc2_input_zp;
extern const float fc2_output_scale;
extern const int32_t fc2_output_zp;

// Conv2D layer configs
static const Conv2DLayer conv1_layer = {
    .weight = conv1_weight,
    .bias = conv1_bias,
    .weight_scale = conv1_weight_scale,
    .weight_zp = conv1_weight_zp,
    .input_scale = 0.0039215686f,  // 1/255
    .input_zp = 0,
    .output_scale = 0.0f,  // filled at runtime
    .output_zp = 0,
    .output_mult = conv1_output_mult,
    .output_shift = conv1_output_shift,
    .in_channels = 1,
    .out_channels = 6,
    .kernel_h = 3,
    .kernel_w = 3,
    .stride = 1,
    .padding = 1,
};

static const Conv2DLayer conv2_layer = {
    .weight = conv2_weight,
    .bias = conv2_bias,
    .weight_scale = conv2_weight_scale,
    .weight_zp = conv2_weight_zp,
    .input_scale = 0.0f,  // filled at runtime (conv1 output scale)
    .input_zp = 0,
    .output_scale = 0.0f,
    .output_zp = 0,
    .output_mult = conv2_output_mult,
    .output_shift = conv2_output_shift,
    .in_channels = 6,
    .out_channels = 12,
    .kernel_h = 3,
    .kernel_w = 3,
    .stride = 1,
    .padding = 1,
};

static const MaxPoolLayer pool1_layer = {
    .kernel_size = 2,
    .stride = 2,
    .padding = 0,
};

static const MaxPoolLayer pool2_layer = {
    .kernel_size = 2,
    .stride = 2,
    .padding = 0,
};

static const FCLayer fc1_layer = {
    .weight = fc1_weight,
    .bias = fc1_bias,
    .weight_scale = fc1_weight_scale,
    .weight_zp = fc1_weight_zp,
    .input_scale = 0.0f,
    .input_zp = 0,
    .output_scale = 0.0f,
    .output_zp = 0,
    .output_mult = fc1_output_mult,
    .output_shift = fc1_output_shift,
    .in_features = 12 * 7 * 7,
    .out_features = 16,
};

static const FCLayer fc2_layer = {
    .weight = fc2_weight,
    .bias = fc2_bias,
    .weight_scale = fc2_weight_scale,
    .weight_zp = fc2_weight_zp,
    .input_scale = 0.0f,
    .input_zp = 0,
    .output_scale = 0.0f,
    .output_zp = 0,
    .output_mult = fc2_output_mult,
    .output_shift = fc2_output_shift,
    .in_features = 16,
    .out_features = 10,
};

int inference_run(const uint8_t* input, uint8_t* output_logits) {
    tensor_arena_init(&g_arena);
    
    // Input: 28x28x1 UINT8
    uint8_t* tensor1 = tensor_arena_alloc_u8(&g_arena, 28 * 28 * 1);
    if (!tensor1) return -1;
    // Copy input (already in correct format)
    for (int i = 0; i < 28 * 28; i++) tensor1[i] = input[i];
    
    // Conv1: 28x28x1 -> 28x28x6
    uint8_t* tensor2 = tensor_arena_alloc_u8(&g_arena, 28 * 28 * 6);
    if (!tensor2) return -1;
    conv2d_run(&conv1_layer, tensor1, 28, 28, tensor2, &g_arena);
    
    // Pool1: 28x28x6 -> 14x14x6
    uint8_t* tensor3 = tensor_arena_alloc_u8(&g_arena, 14 * 14 * 6);
    if (!tensor3) return -1;
    maxpool_run(&pool1_layer, tensor2, 28, 28, 6, tensor3);
    
    // Conv2: 14x14x6 -> 14x14x12
    uint8_t* tensor4 = tensor_arena_alloc_u8(&g_arena, 14 * 14 * 12);
    if (!tensor4) return -1;
    conv2d_run(&conv2_layer, tensor3, 14, 14, tensor4, &g_arena);
    
    // Pool2: 14x14x12 -> 7x7x12
    uint8_t* tensor5 = tensor_arena_alloc_u8(&g_arena, 7 * 7 * 12);
    if (!tensor5) return -1;
    maxpool_run(&pool2_layer, tensor4, 14, 14, 12, tensor5);
    
    // FC1: 588 -> 16
    uint8_t* tensor6 = tensor_arena_alloc_u8(&g_arena, 16);
    if (!tensor6) return -1;
    fc_run(&fc1_layer, tensor5, tensor6, &g_arena);
    
    // FC2: 16 -> 10 (output logits)
    fc_run(&fc2_layer, tensor6, output_logits, &g_arena);
    
    g_peak_arena_usage = tensor_arena_peak_usage(&g_arena);
    return 0;
}

size_t inference_get_peak_arena_usage(void) {
    return g_peak_arena_usage;
}