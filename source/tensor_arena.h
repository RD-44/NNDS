#ifndef TENSOR_ARENA_H
#define TENSOR_ARENA_H

#include <stdint.h>
#include <stddef.h>
#include "fixed_point.h"

// Maximum tensor arena size (defined in model.h)
#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE (24 * 1024)
#endif

// Tensor arena structure
typedef struct {
    uint8_t buffer[TENSOR_ARENA_SIZE];
    size_t offset;
    size_t peak_usage;
} TensorArena;

// Initialize arena
static inline void tensor_arena_init(TensorArena* arena) {
    arena->offset = 0;
    arena->peak_usage = 0;
}

// Allocate aligned memory from arena
static inline void* tensor_arena_alloc(TensorArena* arena, size_t size, size_t alignment) {
    size_t aligned_offset = (arena->offset + alignment - 1) & ~(alignment - 1);
    
    if (aligned_offset + size > TENSOR_ARENA_SIZE) {
        return NULL; // Out of memory
    }
    
    void* ptr = &arena->buffer[aligned_offset];
    arena->offset = aligned_offset + size;
    
    if (arena->offset > arena->peak_usage) {
        arena->peak_usage = arena->offset;
    }
    
    return ptr;
}

// Allocate for UINT8 tensor
static inline uint8_t* tensor_arena_alloc_u8(TensorArena* arena, size_t count) {
    return (uint8_t*)tensor_arena_alloc(arena, count * sizeof(uint8_t), 4);
}

// Allocate for INT8 tensor
static inline int8_t* tensor_arena_alloc_i8(TensorArena* arena, size_t count) {
    return (int8_t*)tensor_arena_alloc(arena, count * sizeof(int8_t), 4);
}

// Allocate for INT32 tensor
static inline int32_t* tensor_arena_alloc_i32(TensorArena* arena, size_t count) {
    return (int32_t*)tensor_arena_alloc(arena, count * sizeof(int32_t), 4);
}

// Reset arena (free all allocations)
static inline void tensor_arena_reset(TensorArena* arena) {
    arena->offset = 0;
}

// Get current usage
static inline size_t tensor_arena_usage(const TensorArena* arena) {
    return arena->offset;
}

// Get peak usage
static inline size_t tensor_arena_peak_usage(const TensorArena* arena) {
    return arena->peak_usage;
}

#endif // TENSOR_ARENA_H