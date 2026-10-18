# NNDS Makefile using devkitPro standard build system

ifeq ($(strip $(DEVKITARM)),)
$(error "Please set DEVKITARM in your environment. export DEVKITARM=<path to>devkitARM")
endif

include $(DEVKITARM)/ds_rules

# Project settings
TARGET   := mnist_ds
BUILD    := build
SOURCES  := source source/layers
INCLUDES := source source/layers
DATA     :=
GRAPHICS :=
AUDIO    :=
ICON     :=

ARCH := -march=armv5te -mtune=arm946e-s

CFLAGS   := -g -Wall -O2 -ffunction-sections -fdata-sections $(ARCH) $(INCLUDE) -DARM9
CXXFLAGS := $(CFLAGS) -fno-rtti -fno-exceptions
ASFLAGS  := -g $(ARCH)
LDFLAGS  := -specs=ds_arm9.specs -g $(ARCH) -Wl,-Map,$(notdir $*.map)

LIBS := -lnds9

LIBDIRS := $(LIBNDS) $(PORTLIBS)

# Python pipeline targets
PYTHON := python3
TOOLS_DIR := tools
CHECKPOINTS_DIR := checkpoints

# Override source detection to include layers subdirectory
CFILES   := $(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.c)))
CPPFILES := $(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.cpp)))
SFILES   := $(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.s)))

# Project root directory (works from both project root and build directory)
PROJECT_ROOT := $(shell pwd)

export OUTPUT := $(PROJECT_ROOT)/$(TARGET)
export VPATH := $(foreach dir,$(SOURCES),$(PROJECT_ROOT)/$(dir))\
                $(foreach dir,$(DATA),$(PROJECT_ROOT)/$(dir))\
                $(foreach dir,$(GRAPHICS),$(PROJECT_ROOT)/$(dir))
export DEPSDIR := $(PROJECT_ROOT)/$(BUILD)

OFILES_SOURCES := $(CPPFILES:.cpp=.o) $(CFILES:.c=.o) $(SFILES:.s=.o)
export OFILES := $(OFILES_SOURCES)
export HFILES := $(OFILES_SOURCES:.o=.h)

export INCLUDE  := $(foreach dir,$(INCLUDES),-iquote $(PROJECT_ROOT)/$(dir))\
                   $(foreach dir,$(LIBDIRS),-I$(dir)/include)\
                   -I$(PROJECT_ROOT)/$(BUILD)
export LIBPATHS := $(foreach dir,$(LIBDIRS),-L$(dir)/lib)

# Use CC for linking C projects
export LD := $(CC)

.PHONY: all train quantize export validate pipeline run clean clean-all $(BUILD)

ifeq ($(BUILD),$(notdir $(CURDIR)))

#---------------------------------------------------------------------------------
# Running from build directory - main build targets
#---------------------------------------------------------------------------------
all: $(OUTPUT).nds

# Python pipeline
train: $(CHECKPOINTS_DIR)/best_ultra_tiny_cnn.pt

quantize: $(CHECKPOINTS_DIR)/quantized_ultra_tiny_cnn.pt

export: $(CURDIR)/source/model.h

validate: test_vectors.npz

pipeline: train quantize export validate

# Full build pipeline
full: pipeline $(OUTPUT).nds

# Python steps
$(CHECKPOINTS_DIR)/best_ultra_tiny_cnn.pt: $(TOOLS_DIR)/train.py $(TOOLS_DIR)/model_def.py
	@mkdir -p $(CHECKPOINTS_DIR)
	cd $(TOOLS_DIR) && $(PYTHON) train.py

$(CHECKPOINTS_DIR)/quantized_ultra_tiny_cnn.pt: $(TOOLS_DIR)/quantize.py $(CHECKPOINTS_DIR)/best_ultra_tiny_cnn.pt
	cd $(TOOLS_DIR) && $(PYTHON) quantize.py

$(PROJECT_ROOT)/source/model.h: $(TOOLS_DIR)/export_model.py $(CHECKPOINTS_DIR)/quantized_ultra_tiny_cnn.pt
	cd $(TOOLS_DIR) && $(PYTHON) export_model.py

test_vectors.npz: $(TOOLS_DIR)/validate.py $(CHECKPOINTS_DIR)/quantized_ultra_tiny_cnn.pt $(CHECKPOINTS_DIR)/best_ultra_tiny_cnn.pt
	cd $(TOOLS_DIR) && $(PYTHON) validate.py

# Run on emulator
run: $(OUTPUT).nds
	melonDS $<

# Clean
clean:
	@echo clean ...
	@rm -fr $(BUILD) $(TARGET).elf $(TARGET).nds

clean-all: clean
	@rm -rf $(CHECKPOINTS_DIR)
	@rm -rf $(TOOLS_DIR)/data
	@rm -f $(CURDIR)/source/model.h
	@rm -f test_vectors.npz

else

#---------------------------------------------------------------------------------
# Running from project root - recursive make
#---------------------------------------------------------------------------------

all: $(BUILD)

$(BUILD):
	@mkdir -p $@
	@$(MAKE) --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile

endif