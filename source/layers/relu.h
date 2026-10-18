#ifndef RELU_H
#define RELU_H

#include <stdint.h>

// In-place ReLU for UINT8 (since UINT8 is already >= 0, this is a no-op for post-ReLU activations)
// But useful if we have INT8 activations
static inline void relu_u8(uint8_t* data, int count) {
    // UINT8 is already non-negative, no-op
    (void)data; (void)count;
}

static inline void relu_i8(int8_t* data, int count) {
    for (int i = 0; i < count; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}

#endif // RELU_H