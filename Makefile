# NNDS Makefile - standalone, no recursive make

DEVKITARM := /opt/devkitpro/devkitARM
DEVKITPRO := /opt/devkitpro

PREFIX := $(DEVKITARM)/bin/arm-none-eabi-
CC := $(PREFIX)gcc
CXX := $(PREFIX)g++
OBJCOPY := $(PREFIX)objcopy
SIZE := $(PREFIX)size
NDSTOOL := ndstool

TARGET := mnist_ds
BUILD_DIR := build
SOURCE_DIR := source
TOOLS_DIR := tools
CHECKPOINTS_DIR := checkpoints

ARCH := -march=armv5te -mtune=arm946e-s

CFLAGS := -mthumb -mthumb-interwork $(ARCH) -O2 -Wall -Wextra -ffunction-sections -fdata-sections \
          -I$(SOURCE_DIR) -I$(DEVKITPRO)/libnds/include -I$(DEVKITPRO)/calico/include -DARM9 -D__NDS__

LDFLAGS := -T$(DEVKITPRO)/calico/lib/ds9.ld -Wl,-Map,$(BUILD_DIR)/$(TARGET).map -Wl,--gc-sections \
           -L$(DEVKITPRO)/libnds/lib -L$(DEVKITPRO)/calico/lib -lnds9 -lcalico_ds9 -lfat -lm

PYTHON := $(CURDIR)/.venv/bin/python3

# Source files
SOURCES := $(SOURCE_DIR)/main.c \
           $(SOURCE_DIR)/inference.c \
           $(SOURCE_DIR)/tensor_arena.c \
           $(SOURCE_DIR)/layers/conv2d.c \
           $(SOURCE_DIR)/layers/maxpool.c \
           $(SOURCE_DIR)/layers/fc.c

OBJECTS := $(SOURCES:$(SOURCE_DIR)/%.c=$(BUILD_DIR)/%.o)

.PHONY: all train quantize export validate pipeline run clean clean-all

all: $(BUILD_DIR)/$(TARGET).nds

# Python pipeline
train: $(CHECKPOINTS_DIR)/best_ultra_tiny_cnn.pt

quantize: $(CHECKPOINTS_DIR)/quantized_ultra_tiny_cnn.pt

export: $(SOURCE_DIR)/model.h

validate: test_vectors.npz

pipeline: train quantize export validate

full: pipeline $(BUILD_DIR)/$(TARGET).nds

# DS build
$(BUILD_DIR)/$(TARGET).nds: $(BUILD_DIR)/$(TARGET).elf
	$(NDSTOOL) -c $@ -9 $< -7 $(DEVKITPRO)/calico/bin/ds7_lykoi.elf -b $(DEVKITPRO)/calico/share/nds-icon.bmp "$(TARGET);built with devkitARM;http://devkitpro.org"

$(BUILD_DIR)/$(TARGET).elf: $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $@"
	@$(SIZE) $@

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Python steps
$(CHECKPOINTS_DIR)/best_ultra_tiny_cnn.pt: $(TOOLS_DIR)/train.py $(TOOLS_DIR)/model_def.py
	@mkdir -p $(CHECKPOINTS_DIR)
	$(PYTHON) $(TOOLS_DIR)/train.py

$(CHECKPOINTS_DIR)/quantized_ultra_tiny_cnn.pt: $(TOOLS_DIR)/quantize.py $(CHECKPOINTS_DIR)/best_ultra_tiny_cnn.pt
	$(PYTHON) $(TOOLS_DIR)/quantize.py

$(SOURCE_DIR)/model.h: $(TOOLS_DIR)/export_model.py $(CHECKPOINTS_DIR)/quantized_ultra_tiny_cnn.pt
	$(PYTHON) $(TOOLS_DIR)/export_model.py

test_vectors.npz: $(TOOLS_DIR)/validate.py $(CHECKPOINTS_DIR)/quantized_ultra_tiny_cnn.pt $(CHECKPOINTS_DIR)/best_ultra_tiny_cnn.pt
	$(PYTHON) $(TOOLS_DIR)/validate.py

run: $(BUILD_DIR)/$(TARGET).nds
	melonDS $<

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(SOURCE_DIR)/model.h
	rm -f test_vectors.npz

clean-all: clean
	rm -rf $(CHECKPOINTS_DIR)
	rm -rf $(TOOLS_DIR)/data