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

// Test vector (first sample from test_vectors.npz - label 7)
static const uint8_t test_input[784] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,  84, 185, 159, 151,  60,  36,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198, 198, 198, 198, 170,  52,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,  67, 114,  72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  17,  66,  14,  67,  67,  67,  59,  21, 236, 254, 106,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  83, 253, 209,  18,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  22, 233, 255,  83,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 129, 254, 238,  44,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  59, 249, 254,  62,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 133, 254, 187,   5,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   9, 205, 248,  58,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 126, 254, 182,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  75, 251, 240,  57,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  19, 221, 254, 166,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3, 203, 254, 219,  35,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  38, 254, 254,  77,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  31, 224, 254, 115,   1,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 133, 254, 254,  52,   0,
    0,   0,   0,   0,   0,   0,   0,   0,  61, 242, 254, 254,  52,   0,
    0,   0,   0,   0,   0,   0, 121, 254, 254, 219,  40,   0,
    0,   0,   0,   0,   0, 121, 254, 207,  18,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
};

// Sprite index constants - dedicated ranges
#define SPRITE_BOX_START    0    // Box border: 0-67 (68 sprites max)
#define SPRITE_DRAW_START   70   // Drawing: 70-127 (58 sprites)
#define SPRITE_TEST_START   70   // Test mode: same as drawing
#define SPRITE_MAX          127  // Max sprite index (128 total)

// Drawing canvas
#define CANVAS_W 256
#define CANVAS_H 192

// Console width is 32 chars (256/8)
#define CONSOLE_WIDTH 32

// Clear a console line
static inline void clear_line(int line) {
    iprintf("\x1b[%d;0H%*s", line, CONSOLE_WIDTH, "");
}

// Clear lines from start to end inclusive
static void clear_lines(int start, int end) {
    for (int i = start; i <= end; i++) clear_line(i);
}

// Clear input buffer
void clear_input() {
    memset(input_buffer, 0, sizeof(input_buffer));
    stroke_count = 0;
    drawing = false;
}

// Draw the persistent box border using fixed sprite indices (SPRITE_BOX_START to SPRITE_BOX_START+67)
static void draw_box_border() {
    int sprite_idx = SPRITE_BOX_START;
    
    for (int i = 0; i < 168; i++) {
        // Top border
        oamSet(&oamMain, sprite_idx, 44 + i, 40, 0, 0,
               SpriteSize_8x8, SpriteColorFormat_16Color, SPRITE_GFX, 0, false, false, false, false, false);
        sprite_idx++;
        // Bottom border
        oamSet(&oamMain, sprite_idx, 44 + i, 40 + 168 - 8, 0, 0,
               SpriteSize_8x8, SpriteColorFormat_16Color, SPRITE_GFX, 0, false, false, false, false, false);
        sprite_idx++;
    }
    for (int i = 0; i < 168; i++) {
        // Left border
        oamSet(&oamMain, sprite_idx, 44, 40 + i, 0, 0,
               SpriteSize_8x8, SpriteColorFormat_16Color, SPRITE_GFX, 0, false, false, false, false, false);
        sprite_idx++;
        // Right border
        oamSet(&oamMain, sprite_idx, 44 + 168 - 8, 40 + i, 0, 0,
               SpriteSize_8x8, SpriteColorFormat_16Color, SPRITE_GFX, 0, false, false, false, false, false);
        sprite_idx++;
    }
}

// Clear drawing sprites (SPRITE_DRAW_START to SPRITE_MAX)
static void clear_drawing_sprites() {
    for (int i = SPRITE_DRAW_START; i <= SPRITE_MAX; i++) {
        oamSet(&oamMain, i, -32, -32, 0, 0, SpriteSize_8x8, SpriteColorFormat_16Color, 
               SPRITE_GFX, 0, false, false, false, false, false);
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
    
    // Save stroke count before clearing
    int saved_stroke_count = stroke_count;
    
    // Clear buffer only (don't reset stroke_count)
    memset(input_buffer, 0, sizeof(input_buffer));
    drawing = false;
    
    // Draw stroke into 28x28 buffer with anti-aliasing (simple)
    for (int i = 1; i < saved_stroke_count; i++) {
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

int main() {
    // Put main screen on bottom LCD (touchscreen)
    lcdMainOnBottom();
    
    // Initialize video - MODE_0_2D for 2D graphics
    videoSetMode(MODE_0_2D);
    
    // VRAM mapping - match official example: A for BG (console), B for sprites
    vramSetBankA(VRAM_A_MAIN_BG);
    vramSetBankB(VRAM_B_MAIN_SPRITE);
    
    // Initialize OAM for sprites
    oamInit(&oamMain, SpriteMapping_1D_128, false);
    
    // Set up sprite palette - index 1 = white (index 0 is transparent)
    SPRITE_PALETTE[1] = RGB15(31, 31, 31);
    
    // Load dot sprite into VRAM_B (16-color format: 4 bits per pixel, 2 pixels per u16)
    // For 8x8 sprite in 16-color mode: 8x8/2 = 32 u16 values
    for (int i = 0; i < 32; i++) {
        SPRITE_GFX[i] = 0x1111;
    }
    
    // Initialize console on main engine (bottom screen after lcdMainOnBottom)
    // Uses VRAM_A for background
    consoleInit(0, 0, BgType_Text4bpp, BgSize_T_256x256, 31, 0, true, true);
    
    // Initialize touchscreen
    touchPosition touch;
    touchRead(&touch);
    
    iprintf("\x1b[1;0HMNIST Digit Recognition on DS");
    iprintf("\x1b[2;0HDraw a digit (0-9) with stylus");
    iprintf("\x1b[3;0HPRESS START to clear, SELECT to infer, L+R to test");
    
    clear_input();
    
    // Draw persistent box border (fixed sprite indices 0-67)
    draw_box_border();
    oamUpdate(&oamMain);
    
    // Drawing sprite index starts at SPRITE_DRAW_START (70)
    int drawing_sprite_idx = SPRITE_DRAW_START;
    
    while (1) {
        swiWaitForVBlank();
        
        // Read keys
        scanKeys();
        int pressed = keysDown();
        int held = keysHeld();
        
        // Read touchscreen
        touchRead(&touch);
        
        // Debug: print keys held and touch state (line 21)
        iprintf("\x1b[21;0HKeys: %08X Touch: %d,%d   ", held, touch.px, touch.py);
        
        if (pressed & KEY_START) {
            clear_input();
            
            // Clear only drawing sprites (keep box border)
            clear_drawing_sprites();
            
            // Drawing sprite index resets
            drawing_sprite_idx = SPRITE_DRAW_START;
            
            // Clear prediction area and redraw prompt
            clear_prediction_area();
            iprintf("\x1b[2;0HDraw a digit (0-9) with stylus");
            oamUpdate(&oamMain);
        }
        
        if (pressed & KEY_SELECT) {
            // Clear inference status line
            clear_line(19);
            iprintf("\x1b[19;0HRunning inference...   ");
            
            iprintf("\x1b[22;0HStroke count: %d   ", stroke_count);
            iprintf("\x1b[23;0HFirst few pts: (%d,%d) (%d,%d) (%d,%d)",
                    stroke_points[0].x, stroke_points[0].y,
                    stroke_points[1].x, stroke_points[1].y,
                    stroke_points[2].x, stroke_points[2].y);
            
            stroke_to_mnist();
            
            // Debug: check input buffer after conversion
            int non_zero = 0;
            for (int i = 0; i < 28*28; i++) if (input_buffer[i] > 0) non_zero++;
            iprintf("\x1b[23;0HInput buffer non-zero: %d   ", non_zero);
            
            int result = inference_run(input_buffer, output_logits);
            
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
            for (int i = 0; i < 784; i++) input_buffer[i] = test_input[i];
            
            // Draw test input using sprites - create a square box (168x168 pixels)
            // 28x28 scaled by 6 = 168x168, centered horizontally, at y=40 to avoid console text
            int test_sprite_idx = SPRITE_TEST_START;
            const int box_size = 168;
            
            // Draw box border using test sprite indices
            int test_box_sprite = SPRITE_TEST_START;
            for (int i = 0; i < 168; i++) {
                if (test_box_sprite > SPRITE_MAX) break;
                // Top border
                oamSet(&oamMain, test_box_sprite, 44 + i, 40, 0, 0,
                       SpriteSize_8x8, SpriteColorFormat_16Color, SPRITE_GFX, 0, false, false, false, false, false);
                test_box_sprite++;
                // Bottom border
                oamSet(&oamMain, test_box_sprite, 44 + i, 40 + 168 - 8, 0, 0,
                       SpriteSize_8x8, SpriteColorFormat_16Color, SPRITE_GFX, 0, false, false, false, false, false);
                test_box_sprite++;
            }
            for (int i = 0; i < 168; i++) {
                if (test_box_sprite > SPRITE_MAX) break;
                // Left border
                oamSet(&oamMain, test_box_sprite, 44, 40 + i, 0, 0,
                       SpriteSize_8x8, SpriteColorFormat_16Color, SPRITE_GFX, 0, false, false, false, false, false);
                test_box_sprite++;
                // Right border
                oamSet(&oamMain, test_box_sprite, 44 + 168 - 8, 40 + i, 0, 0,
                       SpriteSize_8x8, SpriteColorFormat_16Color, SPRITE_GFX, 0, false, false, false, false, false);
                test_box_sprite++;
            }
            
            // Draw test input pixels inside the box (scale 28->168 = 6x)
            for (int y = 0; y < 28 && test_sprite_idx <= SPRITE_MAX; y++) {
                for (int x = 0; x < 28 && test_sprite_idx <= SPRITE_MAX; x++) {
                    if (test_input[y * 28 + x] > 128) {
                        int px = 44 + x * 6 + 3;  // Center in 6x6 cell
                        int py = 40 + y * 6 + 3;
                        oamSet(&oamMain, test_sprite_idx,
                               px, py, 0, 0,
                               SpriteSize_8x8, SpriteColorFormat_16Color,
                               SPRITE_GFX, 0,
                               false, false, false, false, false);
                        test_sprite_idx++;
                    }
                }
            }
            oamUpdate(&oamMain);
            
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
            iprintf("\x1b[19;0HTest: pred=%d (exp 7) conf=%.1f%%  ", max_idx, conf * 100);
            draw_prediction(max_idx, conf);
        }
        
        // Handle drawing - check KEY_TOUCH and raw coordinates
        if (held & KEY_TOUCH && touch.rawx != 0 && touch.rawy != 0) {
            if (!drawing) {
                drawing = true;
                stroke_count = 0;
                
                // Clear prediction area when starting new drawing
                clear_prediction_area();
            }
            
            if (stroke_count < MAX_STROKE_POINTS) {
                stroke_points[stroke_count].x = touch.px;
                stroke_points[stroke_count].y = touch.py;
                stroke_points[stroke_count].active = true;
                stroke_count++;
            }
            
            // Draw line from previous point to current point using sprites (skip pixels to save sprites)
            if (stroke_count > 1 && drawing_sprite_idx <= SPRITE_MAX) {
                int x0 = stroke_points[stroke_count-2].x;
                int y0 = stroke_points[stroke_count-2].y;
                int x1 = touch.px;
                int y1 = touch.py;
                
                // Bresenham line - place sprites along the line (every 4th pixel)
                int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
                int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
                int err = dx + dy, e2;
                int step = 0;
                
                while (true) {
                    if (drawing_sprite_idx > SPRITE_MAX) break;
                    if (step % 4 == 0) {  // Only draw every 4th pixel
                        oamSet(&oamMain, drawing_sprite_idx,
                               x0 - 4, y0 - 4,
                               0, 0,
                               SpriteSize_8x8, SpriteColorFormat_16Color,
                               SPRITE_GFX, 0,
                               false, false, false, false, false);
                        drawing_sprite_idx++;
                    }
                    
                    if (x0 == x1 && y0 == y1) break;
                    e2 = 2 * err;
                    if (e2 >= dy) { err += dy; x0 += sx; }
                    if (e2 <= dx) { err += dx; y0 += sy; }
                    step++;
                }
            }
            
            // Debug: print touch coordinates
            iprintf("\x1b[20;0HTouch: px=%3d py=%3d rawx=%4d rawy=%4d count=%3d sprite=%3d   ", 
                    touch.px, touch.py, touch.rawx, touch.rawy, stroke_count, drawing_sprite_idx);
        } else {
            drawing = false;
            clear_line(20);
            iprintf("\x1b[20;0HNo touch detected");
        }
        
        // Update sprites (this updates both box border and drawing)
        oamUpdate(&oamMain);
    }
    
    return 0;
}