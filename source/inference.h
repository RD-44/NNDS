#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "tensor_arena.h"

// Run full inference: input[784] UINT8 -> output[10] UINT8 (logits)
// Returns predicted digit (0-9)
int inference_run(const uint8_t* input, uint8_t* output_logits);

// Get tensor arena peak usage for debugging
size_t inference_get_peak_arena_usage(void);

#endif // INFERENCE_H