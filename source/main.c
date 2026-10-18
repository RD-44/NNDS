#include <nds.h>
#include <stdio.h>
#include <string.h>
#include "inference.h"
#include "tensor_arena.h"

// Touchscreen coordinates
typedef struct {
    int x, y;
    bool active;
} TouchPoint;

#define MAX_STROKE_POINTS 256
static TouchPoint stroke_points[MAX_STROKE_POINTS];
static int stroke_count = 0;
static bool drawing = false;

// 28x28 input buffer (UINT8)
static uint8_t input_buffer[28 * 28];
static uint8_t output_logits[10];

// Drawing canvas (256x192 on bottom screen)
#define CANVAS_W 256
#define CANVAS_H 192
#define CANVAS_X 0
#define CANVAS_Y 0

// Clear input buffer
void clear_input() {
    memset(input_buffer, 0, sizeof(input_buffer));
    stroke_count = 0;
    drawing = false;
}

// Convert stroke points to 28x28 centered image
void stroke_to_mnist() {
    if (stroke_count < 2) return;
    
    // Find bounding box
    int min_x = CANVAS_W, max_x = 0;
    int min_y = CANVAS_H, max_y = 0;
    
    for (int i = 0; i < stroke_count; i++) {
        if (stroke_points[i].x < min_x) min_x = stroke_points[i].x;
        if (stroke_points[i].x > max_x) max_x = stroke_points[i].x;
        if (stroke_points[i].y < min_y) min_y = stroke_points[i].y;
        if (stroke_points[i].y > max_y) max_y = stroke_points[i].y;
    }
    
    int stroke_w = max_x - min_x;
    int stroke_h = max_y - min_y;
    if (stroke_w == 0) stroke_w = 1;
    if (stroke_h == 0) stroke_h = 1;
    
    // Scale to fit in 20x20 box, centered in 28x28
    const int target_size = 20;
    float scale_x = (float)target_size / stroke_w;
    float scale_y = (float)target_size / stroke_h;
    float scale = (scale_x < scale_y) ? scale_x : scale_y;
    
    int scaled_w = (int)(stroke_w * scale);
    int scaled_h = (int)(stroke_h * scale);
    int offset_x = (28 - scaled_w) / 2;
    int offset_y = (28 - scaled_h) / 2;
    
    // Clear buffer
    clear_input();
    
    // Draw stroke into 28x28 buffer with anti-aliasing (simple)
    for (int i = 1; i < stroke_count; i++) {
        int x0 = (int)((stroke_points[i-1].x - min_x) * scale) + offset_x;
        int y0 = (int)((stroke_points[i-1].y - min_y) * scale) + offset_y;
        int x1 = (int)((stroke_points[i].x - min_x) * scale) + offset_x;
        int y1 = (int)((stroke_points[i].y - min_y) * scale) + offset_y;
        
        // Bresenham line
        int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2;
        
        while (true) {
            if (x0 >= 0 && x0 < 28 && y0 >= 0 && y0 < 28) {
                input_buffer[y0 * 28 + x0] = 255;
                // Thicken stroke slightly
                if (x0 > 0) input_buffer[y0 * 28 + x0 - 1] = 255;
                if (x0 < 27) input_buffer[y0 * 28 + x0 + 1] = 255;
                if (y0 > 0) input_buffer[(y0-1) * 28 + x0] = 255;
                if (y0 < 27) input_buffer[(y0+1) * 28 + x0] = 255;
            }
            if (x0 == x1 && y0 == y1) break;
            e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }
}

// Draw stroke on screen
void draw_stroke_on_screen() {
    for (int i = 1; i < stroke_count; i++) {
        int x0 = stroke_points[i-1].x;
        int y0 = stroke_points[i-1].y;
        (void)stroke_points[i].x;
        (void)stroke_points[i].y;
        
        // Draw thick line
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int px = x0 + dx;
                int py = y0 + dy;
                if (px >= 0 && px < CANVAS_W && py >= 0 && py < CANVAS_H) {
                    VRAM_A[py * CANVAS_W + px] = RGB15(31, 31, 31);
                }
            }
        }
    }
}

// Draw prediction result
void draw_prediction(int digit, float confidence) {
    // Clear right side
    for (int y = 0; y < CANVAS_H; y++) {
        for (int x = 200; x < CANVAS_W; x++) {
            VRAM_A[y * CANVAS_W + x] = RGB15(0, 0, 0);
        }
    }
    
    // Draw large digit
    char buf[32];
    sprintf(buf, "Prediction: %d", digit);
    iprintf("\x1b[2;0H%s", buf);
    
    sprintf(buf, "Confidence: %.1f%%", confidence * 100);
    iprintf("\x1b[3;0H%s", buf);
    
    // Draw all logits as bars
    iprintf("\x1b[5;0HLogits:");
    for (int i = 0; i < 10; i++) {
        int bar_len = (output_logits[i] * 20) / 255;
        char bar[32] = "";
        for (int j = 0; j < bar_len; j++) strcat(bar, "#");
        sprintf(buf, "%d: %-20s %3d", i, bar, output_logits[i]);
        iprintf("\x1b[%d;0H%s", 6 + i, buf);
    }
    
    // Arena usage
    sprintf(buf, "Arena: %zu / %d bytes", inference_get_peak_arena_usage(), TENSOR_ARENA_SIZE);
    iprintf("\x1b[17;0H%s", buf);
}

int main() {
    // Initialize video
    videoSetMode(MODE_5_2D);
    vramSetBankA(VRAM_A_MAIN_BG);
    
    // Initialize console on bottom screen
    consoleDemoInit();
    
    // Initialize touchscreen
    touchPosition touch;
    
    iprintf("\x1b[1;0HMNIST Digit Recognition on DS");
    iprintf("\x1b[2;0HDraw a digit (0-9) with stylus");
    iprintf("\x1b[3;0HPRESS START to clear, SELECT to infer");
    
    clear_input();
    
    while (1) {
        scanKeys();
        int keys = keysDown();
        
        // Read touch
        touchRead(&touch);
        
        if (keys & KEY_START) {
            clear_input();
            // Clear screen
            for (int y = 0; y < CANVAS_H; y++) {
                for (int x = 0; x < CANVAS_W; x++) {
                    VRAM_A[y * CANVAS_W + x] = RGB15(0, 0, 0);
                }
            }
            iprintf("\x1b[2;0HCleared. Draw a digit (0-9)");
        }
        
        if (keys & KEY_SELECT) {
            stroke_to_mnist();
            inference_run(input_buffer, output_logits);
            
            // Find max logit
            int max_idx = 0;
            uint8_t max_val = output_logits[0];
            for (int i = 1; i < 10; i++) {
                if (output_logits[i] > max_val) {
                    max_val = output_logits[i];
                    max_idx = i;
                }
            }
            
            // Softmax approximation for confidence
            float sum = 0;
            for (int i = 0; i < 10; i++) sum += output_logits[i];
            float conf = (sum > 0) ? (float)max_val / sum : 0;
            
            draw_prediction(max_idx, conf);
        }
        
        // Handle drawing
        if (touch.px < 256 && touch.py < 192) {
            if (!drawing) {
                drawing = true;
                stroke_count = 0;
            }
            
            if (stroke_count < MAX_STROKE_POINTS) {
                stroke_points[stroke_count].x = touch.px;
                stroke_points[stroke_count].y = touch.py;
                stroke_points[stroke_count].active = true;
                stroke_count++;
            }
            
            // Draw current point
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    int px = touch.px + dx;
                    int py = touch.py + dy;
                    if (px >= 0 && px < CANVAS_W && py >= 0 && py < CANVAS_H) {
                        VRAM_A[py * CANVAS_W + px] = RGB15(31, 31, 31);
                    }
                }
            }
        } else {
            drawing = false;
        }
        
        swiWaitForVBlank();
    }
    
    return 0;
}