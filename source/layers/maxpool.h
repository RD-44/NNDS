#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <stdint.h>

typedef struct {
    int kernel_size;
    int stride;
    int padding;
} MaxPoolLayer;

int maxpool_run(const MaxPoolLayer* layer,
                const uint8_t* input, int H, int W, int C,
                uint8_t* output);

#endif // MAXPOOL_H