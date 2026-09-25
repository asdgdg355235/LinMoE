# Native Linux CPU/reference baseline. The existing Windows/CUDA manual build
# remains separate. HIP must not silently fall back to CPU when requested.
CC = cc
CXX = c++
BUILD ?= build/linux-cpu
BACKEND ?= cpu
ifneq ($(BACKEND),cpu)
$(error BACKEND=$(BACKEND) is not implemented yet; only the CPU baseline is available)
endif
CPPFLAGS += -D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64
CFLAGS ?= -O2 -g
CXXFLAGS ?= -O2 -g
# Preserve the existing kernels: several float paths require AVX512F/BW/DQ/VL.
# This is an explicit CPU baseline requirement, independent of the AMD GPU target.
CPU_FLAGS = -mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512dq -mavx512vl -fopenmp
LDLIBS += -lm -fopenmp
RUNTIME = engine/runtime
PLATFORM = engine/platform
HEADERS = $(wildcard $(RUNTIME)/*.h) $(wildcard $(PLATFORM)/*.h)

.PHONY: all check clean
all: $(BUILD)/linmoe $(BUILD)/linmoe-inspect

$(BUILD):
	mkdir -p $@

$(BUILD)/linmoe: $(RUNTIME)/winmoe_inference.c $(RUNTIME)/gpu_offload_cpu.c $(PLATFORM)/posix_io.c $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $(CPU_FLAGS) $(filter %.c,$^) -o $@ $(LDLIBS)

$(BUILD)/linmoe-inspect: tests/gguf_inspect.c $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $< -o $@ -lm

$(BUILD)/q6k-parity: $(RUNTIME)/test/q6k_parity.cpp $(RUNTIME)/q6k_dequant.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 $(CPU_FLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/posix-io-test: tests/posix_io_test.c $(PLATFORM)/posix_io.c $(PLATFORM)/posix_io.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 -Wall -Wextra -Werror $(filter %.c,$^) -o $@

$(BUILD)/q8-dot-test: tests/q8_dot_test.c $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $(CPU_FLAGS) $< -o $@ $(LDLIBS)

# No downloaded model or accelerator is needed for the foundation tests.
check: all $(BUILD)/q6k-parity $(BUILD)/posix-io-test $(BUILD)/q8-dot-test
	$(BUILD)/q6k-parity
	$(BUILD)/q8-dot-test
	$(BUILD)/posix-io-test
	python3 tests/gguf_test.py $(BUILD)/linmoe-inspect
	python3 tests/inference_smoke.py $(BUILD)/linmoe
	$(BUILD)/linmoe --help

# Restrict cleanup to the selected build directory; never touch model data.
clean:
	rm -f $(BUILD)/linmoe $(BUILD)/linmoe-inspect $(BUILD)/q6k-parity $(BUILD)/posix-io-test $(BUILD)/q8-dot-test
