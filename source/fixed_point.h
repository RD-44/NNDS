#ifndef FIXED_POINT_H
#define FIXED_POINT_H

#include <stdint.h>
#include <stddef.h>

// Saturating clamp to uint8_t
static inline uint8_t saturate_u8(int32_t x) {
    if (x > 255) return 255;
    if (x < 0) return 0;
    return (uint8_t)x;
}

// Saturating clamp to int8_t
static inline int8_t saturate_i8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

// Q31 fixed-point multiply: (a * b) >> 31
// a: INT32, b: INT32 (Q31 format)
// Returns INT32
static inline int32_t q31_mul(int32_t a, int32_t b) {
    int64_t result = (int64_t)a * b;
    return (int32_t)(result >> 31);
}

// Q31 multiply with separate shift (for flexibility)
// Returns INT32
static inline int32_t q31_mul_shift(int32_t a, int32_t b, int shift) {
    int64_t result = (int64_t)a * b;
    return (int32_t)(result >> shift);
}

// Requantize INT32 accumulator to UINT8 output
// acc: INT32 accumulator value
// mult: Q31 multiplier (precomputed: scale * 2^31)
// shift: usually 31
// zp: output zero point
// Returns UINT8
static inline uint8_t requantize_u8(int32_t acc, int32_t mult, int shift, int32_t zp) {
    int64_t tmp = (int64_t)acc * mult + ((int64_t)1 << (shift - 1));
    int32_t requantized = (int32_t)(tmp >> shift) + zp;
    return saturate_u8(requantized);
}

// Requantize INT32 accumulator to INT8 output
static inline int8_t requantize_i8(int32_t acc, int32_t mult, int shift, int32_t zp) {
    int64_t tmp = (int64_t)acc * mult + ((int64_t)1 << (shift - 1));
    int32_t requantized = (int32_t)(tmp >> shift) + zp;
    return saturate_i8(requantized);
}

// Multiply INT8 * UINT8 -> INT32 (for accumulation)
static inline int32_t mul_i8_u8(int8_t a, uint8_t b) {
    return (int32_t)a * (int32_t)b;
}

// Multiply INT8 * INT8 -> INT32
static inline int32_t mul_i8_i8(int8_t a, int8_t b) {
    return (int32_t)a * (int32_t)b;
}

#endif // FIXED_POINT_H