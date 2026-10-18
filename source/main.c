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
static int last_px = 0, last_py = 0;
static bool touch_locked = false;

// 28x28 input buffer (UINT8)
static uint8_t input_buffer[28 * 28];
static uint8_t output_logits[10];

// Test vector (first sample from test_vectors.npz - label 7)
static const uint8_t test_input[784] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,  84, 185, 159, 151,  60,  36,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198, 198, 198, 198, 198, 170,  52,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,  67, 114,  72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  17,  66,  14,  67,  67,  67,  59,  21, 236, 254, 106,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  83, 253, 209,  18,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  22, 233, 255,  83,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 129, 254, 238,  44,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  59, 249, 254,  62,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 133, 254, 187,   5,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   9, 205, 248,  58,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 126, 254, 182,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  75, 251, 240,  57,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  19, 221, 254, 166,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3, 203, 254, 219,  35,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  38, 254, 254,  77,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  31, 224, 254, 115,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 133, 254, 254,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  61, 242, 254, 254,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121, 254, 254, 219,  40,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121, 254, 207,  18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
};

// Drawing canvas
#define CANVAS_W 256
#define CANVAS_H 192

// Drawing box: the interior is exactly the 28x28 MNIST input upscaled by
// MNIST_SCALE screen pixels per MNIST pixel, so drawn proportions match
// the training data with no rescaling
#define MNIST_SCALE 5
#define BOX_SIZE (28 * MNIST_SCALE + 2)
#define BOX_X ((CANVAS_W - BOX_SIZE) / 2)
#define BOX_Y ((CANVAS_H - BOX_SIZE) / 2)

// Console width is 32 chars (256/8)
#define CONSOLE_WIDTH 32

// Clear a console line
static inline void clear_line(int line) {
    iprintf("\x1b[%d;0H%*s", line, CONSOLE_WIDTH, "");
}

// Clear input buffer
void clear_input() {
    memset(input_buffer, 0, sizeof(input_buffer));
    stroke_count = 0;
    drawing = false;
}

// Draw the persistent box border
void draw_box_border() {
    // Top and bottom borders
    for (int x = 0; x < BOX_SIZE; x++) {
        BG_GFX[BOX_Y * 256 + BOX_X + x] = ARGB16(1, 31, 31, 31);
        BG_GFX[(BOX_Y + BOX_SIZE - 1) * 256 + BOX_X + x] = ARGB16(1, 31, 31, 31);
    }
    // Left and right borders
    for (int y = 0; y < BOX_SIZE; y++) {
        BG_GFX[(BOX_Y + y) * 256 + BOX_X] = ARGB16(1, 31, 31, 31);
        BG_GFX[(BOX_Y + y) * 256 + BOX_X + BOX_SIZE - 1] = ARGB16(1, 31, 31, 31);
    }
}

// Clear the whole drawing box (border included), then redraw the border
static void clear_drawing_area() {
    for (int y = BOX_Y; y < BOX_Y + BOX_SIZE; y++) {
        for (int x = BOX_X; x < BOX_X + BOX_SIZE; x++) {
            BG_GFX[y * 256 + x] = 0;
        }
    }
    draw_box_border();
}

// Stamp the 5x5 brush at (px, py), staying inside the box border
static void plot_brush(int px, int py) {
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int x = px + dx;
            int y = py + dy;
            if (x >= BOX_X + 1 && x < BOX_X + BOX_SIZE - 1 &&
                y >= BOX_Y + 1 && y < BOX_Y + BOX_SIZE - 1) {
                BG_GFX[y * 256 + x] = ARGB16(1, 31, 31, 31);
            }
        }
    }
}

// Brush a line from (x0,y0) to (x1,y1) so fast strokes don't leave gaps
static void draw_brush_line(int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    
    while (true) {
        plot_brush(x0, y0);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// Clear prediction area (lines 2-17)
static void clear_prediction_area() {
    for (int i = 2; i <= 17; i++) clear_line(i);
}

// Draw prediction result
void draw_prediction(int digit, float confidence) {
    char buf[32];
    
    // Line 2: Prediction
    sprintf(buf, "Prediction: %d", digit);
    iprintf("\x1b[2;0H%s", buf);
    
    // Line 3: Confidence
    sprintf(buf, "Confidence: %.1f%%", confidence * 100);
    iprintf("\x1b[3;0H%s", buf);
    
    // Line 5: Logits header
    iprintf("\x1b[5;0HLogits:");
    
    // Lines 6-15: Logit bars (one per digit)
    for (int i = 0; i < 10; i++) {
        int bar_len = (output_logits[i] * 20) / 255;
        char bar[32] = "";
        for (int j = 0; j < bar_len; j++) strcat(bar, "#");
        sprintf(buf, "%d: %-20s %3d", i, bar, output_logits[i]);
        iprintf("\x1b[%d;0H%s", 6 + i, buf);
    }
    
    // Line 17: Arena usage
    sprintf(buf, "Arena: %zu / %d bytes", inference_get_peak_arena_usage(), TENSOR_ARENA_SIZE);
    iprintf("\x1b[17;0H%s", buf);
}

// Set an MNIST cell to at least the given intensity (keeps the strongest value)
static void mnist_set(int mx, int my, uint8_t v) {
    if (mx < 0 || mx >= 28 || my < 0 || my >= 28) return;
    uint8_t* p = &input_buffer[my * 28 + mx];
    if (*p < v) *p = v;
}

// Plot one screen-space point: a 255 core cell with 128 edge cells, mimicking
// the anti-aliased gray edges of MNIST training digits
static void mnist_plot_point(int sx, int sy) {
    int mx = (sx - (BOX_X + 1)) / MNIST_SCALE;
    int my = (sy - (BOX_Y + 1)) / MNIST_SCALE;
    
    mnist_set(mx, my, 255);
    mnist_set(mx - 1, my, 128);
    mnist_set(mx + 1, my, 128);
    mnist_set(mx, my - 1, 128);
    mnist_set(mx, my + 1, 128);
}

// Convert the drawn strokes into the 28x28 input buffer. The box interior maps
// 1:1 onto the MNIST grid (MNIST_SCALE screen px per cell), so one brush dot
// turns on exactly one MNIST pixel, proportionally identical to the training data
void stroke_to_mnist() {
    if (stroke_count < 1) return;
    
    // Clear buffer only (don't reset stroke_count)
    memset(input_buffer, 0, sizeof(input_buffer));
    drawing = false;
    
    for (int i = 0; i < stroke_count; i++) {
        const int x1 = stroke_points[i].x;
        const int y1 = stroke_points[i].y;
        
        if (i > 0) {
            int x0 = stroke_points[i-1].x;
            int y0 = stroke_points[i-1].y;
            
            // Bresenham in screen space between consecutive touch samples
            int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
            int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
            int err = dx + dy, e2;
            
            while (x0 != x1 || y0 != y1) {
                mnist_plot_point(x0, y0);
                e2 = 2 * err;
                if (e2 >= dy) { err += dy; x0 += sx; }
                if (e2 <= dx) { err += dx; y0 += sy; }
            }
        }
        
        mnist_plot_point(x1, y1);
    }
}

int main() {
    // Put main screen on bottom LCD (touchscreen)
    lcdMainOnBottom();
    
    // Main engine (bottom screen): 16-bit bitmap background for drawing
    videoSetMode(MODE_5_2D);
    vramSetBankA(VRAM_A_MAIN_BG);
    bgInit(3, BgType_Bmp16, BgSize_B16_256x256, 0, 0);
    
    // Console on sub engine (top screen)
    consoleDemoInit();
    
    // Initialize touchscreen
    touchPosition touch;
    touchRead(&touch);
    
    iprintf("\x1b[1;0HMNIST Digit Recognition on DS");
    iprintf("\x1b[2;0HDraw a digit (0-9) with stylus");
    iprintf("\x1b[3;0HSTART:clear SELECT:run L+R:test");
    
    clear_input();
    
    // Clear whole bitmap canvas
    for (int y = 0; y < 192; y++) {
        for (int x = 0; x < 256; x++) {
            BG_GFX[y * 256 + x] = 0;
        }
    }
    
    // Draw persistent box border
    draw_box_border();
    
    while (1) {
        swiWaitForVBlank();
        
        // Read keys
        scanKeys();
        int pressed = keysDown();
        int held = keysHeld();
        
        // Read touchscreen
        touchRead(&touch);
        
        if (pressed & KEY_START) {
            clear_input();
            touch_locked = true;
            
            clear_drawing_area();
            
            // Clear prediction area and status, redraw prompt
            clear_prediction_area();
            clear_line(19);
            iprintf("\x1b[2;0HDraw a digit (0-9) with stylus");
            
        }
        
        if (pressed & KEY_SELECT) {
            // Clear inference status line
            clear_line(19);
            iprintf("\x1b[19;0HRunning inference...   ");
            
            stroke_to_mnist();
            
            inference_run(input_buffer, output_logits);
            
            // Clear inference status
            clear_line(19);
            
            int max_idx = 0;
            uint8_t max_val = output_logits[0];
            for (int i = 1; i < 10; i++) {
                if (output_logits[i] > max_val) {
                    max_val = output_logits[i];
                    max_idx = i;
                }
            }
            
            float sum = 0;
            for (int i = 0; i < 10; i++) sum += output_logits[i];
            float conf = (sum > 0) ? (float)max_val / sum : 0;
            
            draw_prediction(max_idx, conf);
        }
        
        // Test mode: L + R buttons to run test vector
        if ((held & KEY_L) && (held & KEY_R)) {
            clear_line(19);
            iprintf("\x1b[19;0HRunning test vector...   ");
            
            // Clear screen and draw test input visually
            clear_input();
            touch_locked = true;
            for (int i = 0; i < 784; i++) input_buffer[i] = test_input[i];
            
            // Clear drawing area (keep box border)
            clear_drawing_area();
            
            // Draw test input: the exact 28x28 network input, one MNIST pixel
            // per MNIST_SCALE x MNIST_SCALE cell, shaded by its intensity so
            // anti-aliased (light gray) pixels stay visible
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    uint8_t v = test_input[y * 28 + x];
                    if (v == 0) continue;
                    int shade = (v + 7) / 8;
                    if (shade > 31) shade = 31;
                    uint16_t color = ARGB16(1, shade, shade, shade);
                    int px = BOX_X + 1 + x * MNIST_SCALE;
                    int py = BOX_Y + 1 + y * MNIST_SCALE;
                    for (int dy = 0; dy < MNIST_SCALE; dy++) {
                        for (int dx = 0; dx < MNIST_SCALE; dx++) {
                            BG_GFX[(py + dy) * 256 + px + dx] = color;
                        }
                    }
                }
            }
            
            inference_run(test_input, output_logits);
            
            int max_idx = 0;
            uint8_t max_val = output_logits[0];
            for (int i = 1; i < 10; i++) {
                if (output_logits[i] > max_val) {
                    max_val = output_logits[i];
                    max_idx = i;
                }
            }
            
            float sum = 0;
            for (int i = 0; i < 10; i++) sum += output_logits[i];
            float conf = (sum > 0) ? (float)max_val / sum : 0;
            
            clear_line(19);
            iprintf("\x1b[19;0HTest: pred=%d (exp 7) conf=%.1f%%", max_idx, conf * 100);
            draw_prediction(max_idx, conf);
        }
        
        // Handle drawing - check KEY_TOUCH and raw coordinates
        if (held & KEY_TOUCH && touch.rawx != 0 && touch.rawy != 0) {
            // Ignore the stylus until it is lifted if a clear just happened,
            // otherwise a still-held stylus would paint right after clearing
            if (!touch_locked) {
                // Clamp the brush center to the drawable interior
                int cx = touch.px;
                int cy = touch.py;
                if (cx < BOX_X + 1) cx = BOX_X + 1;
                if (cx > BOX_X + BOX_SIZE - 2) cx = BOX_X + BOX_SIZE - 2;
                if (cy < BOX_Y + 1) cy = BOX_Y + 1;
                if (cy > BOX_Y + BOX_SIZE - 2) cy = BOX_Y + BOX_SIZE - 2;
                
                if (!drawing) {
                    drawing = true;
                    stroke_count = 0;
                    last_px = cx;
                    last_py = cy;
                    
                    // Clear prediction area when starting new drawing
                    clear_prediction_area();
                }
                
                if (stroke_count < MAX_STROKE_POINTS) {
                    stroke_points[stroke_count].x = cx;
                    stroke_points[stroke_count].y = cy;
                    stroke_points[stroke_count].active = true;
                    stroke_count++;
                }
                
                // Connect the previous sample to the current one so fast strokes have no gaps
                draw_brush_line(last_px, last_py, cx, cy);
                last_px = cx;
                last_py = cy;
            }
        } else {
            drawing = false;
            touch_locked = false;
        }
    }
    
    return 0;
}