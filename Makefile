# Native Linux build with an explicit CPU or HIP backend. Ordinary runtime
# sources remain C; only HIP translation units and final HIP linkage use hipcc.
CC = cc
CXX = c++
HIPCC ?= hipcc
BACKEND ?= cpu
GPU_ARCH ?= gfx1030
BUILD ?= build/linux-$(BACKEND)

CPPFLAGS += -D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64
CFLAGS ?= -O2 -g
CXXFLAGS ?= -O2 -g
HIPFLAGS ?= -O2 -g
# Preserve the existing kernels: several float paths require AVX512F/BW/DQ/VL.
# This is an explicit CPU baseline requirement, independent of the AMD GPU target.
CPU_FLAGS = -mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512dq -mavx512vl -fopenmp
HIP_ARCH_FLAGS ?= --offload-arch=$(GPU_ARCH)
RUNTIME = engine/runtime
PLATFORM = engine/platform
HEADERS = $(wildcard $(RUNTIME)/*.h) $(wildcard $(PLATFORM)/*.h)

COMMON_OBJS = $(BUILD)/winmoe_inference.o $(BUILD)/posix_io.o

ifeq ($(BACKEND),cpu)
GPU_OBJ = $(BUILD)/gpu_offload_cpu.o
LINK = $(CC)
LINK_FLAGS = -fopenmp
else ifeq ($(BACKEND),hip)
GPU_OBJ = $(BUILD)/gpu_offload_hip.o
LINK = $(HIPCC)
# Runtime C objects are compiled with GCC OpenMP and therefore reference libgomp.
# hipcc performs HIP device/runtime linkage while -lgomp satisfies those host objects.
LINK_FLAGS = $(HIP_ARCH_FLAGS) -lgomp
else
$(error Unsupported BACKEND=$(BACKEND); expected cpu or hip)
endif

.PHONY: all check hip-check hip-toolchain-check clean
all: $(BUILD)/linmoe $(BUILD)/linmoe-inspect

$(BUILD):
	mkdir -p $@

$(BUILD)/winmoe_inference.o: $(RUNTIME)/winmoe_inference.c $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $(CPU_FLAGS) -c $< -o $@

$(BUILD)/posix_io.o: $(PLATFORM)/posix_io.c $(PLATFORM)/posix_io.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 -c $< -o $@

$(BUILD)/gpu_offload_cpu.o: $(RUNTIME)/gpu_offload_cpu.c $(RUNTIME)/gpu_offload.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 -c $< -o $@

ifeq ($(BACKEND),hip)
hip-toolchain-check:
	@command -v "$(HIPCC)" >/dev/null 2>&1 || { \
		echo "LinMoE: HIP compiler '$(HIPCC)' not found. Install ROCm/HIP or set HIPCC=/path/to/hipcc." >&2; \
		exit 127; \
	}

$(BUILD)/gpu_offload_hip.o: $(RUNTIME)/gpu_offload_hip.cpp $(RUNTIME)/gpu_offload.h $(RUNTIME)/gpu_offload_hip_test.h | $(BUILD) hip-toolchain-check
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) -std=c++17 $(HIP_ARCH_FLAGS) -c $< -o $@
endif

$(BUILD)/linmoe: $(COMMON_OBJS) $(GPU_OBJ)
	$(LINK) $^ -o $@ $(LINK_FLAGS) -lm

$(BUILD)/linmoe-inspect: tests/gguf_inspect.c $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $< -o $@ -lm

$(BUILD)/q6k-parity: $(RUNTIME)/test/q6k_parity.cpp $(RUNTIME)/q6k_dequant.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 $(CPU_FLAGS) $< -o $@ -lm -fopenmp

$(BUILD)/posix-io-test: tests/posix_io_test.c $(PLATFORM)/posix_io.c $(PLATFORM)/posix_io.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 -Wall -Wextra -Werror $(filter %.c,$^) -o $@

$(BUILD)/q8-dot-test: tests/q8_dot_test.c $(HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $(CPU_FLAGS) $< -o $@ -lm -fopenmp

# The regression suite is deliberately CPU-only. It remains usable on hosts
# without ROCm and protects the already-verified Linux reference implementation.
ifeq ($(BACKEND),cpu)
check: all $(BUILD)/q6k-parity $(BUILD)/posix-io-test $(BUILD)/q8-dot-test
	$(BUILD)/q6k-parity
	$(BUILD)/q8-dot-test
	$(BUILD)/posix-io-test
	python3 tests/gguf_test.py $(BUILD)/linmoe-inspect
	python3 tests/inference_smoke.py $(BUILD)/linmoe
	$(BUILD)/linmoe --help
else
check:
	@echo "LinMoE: CPU regression suite requires BACKEND=cpu; run 'make BACKEND=cpu check'." >&2
	@echo "LinMoE: accelerator validation is 'make BACKEND=hip GPU_ARCH=$(GPU_ARCH) hip-check'." >&2
	@exit 2
endif

ifeq ($(BACKEND),hip)
# hipcc adds HIP language mode when compiling a .cpp source. Compile each test
# separately so that a following ELF .o is never reinterpreted as HIP source.
$(BUILD)/hip-smoke-test.o: tests/hip_smoke_test.cpp $(RUNTIME)/gpu_offload.h $(RUNTIME)/gpu_offload_hip_test.h | $(BUILD) hip-toolchain-check
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) -std=c++17 $(HIP_ARCH_FLAGS) -c $< -o $@

$(BUILD)/hip-smoke-test: $(BUILD)/hip-smoke-test.o $(BUILD)/gpu_offload_hip.o
	$(HIPCC) $(HIP_ARCH_FLAGS) $^ -o $@

$(BUILD)/hip-q8-parity.o: tests/hip_q8_parity.cpp $(RUNTIME)/gpu_offload.h $(RUNTIME)/gpu_offload_hip_test.h | $(BUILD) hip-toolchain-check
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) -std=c++17 $(HIP_ARCH_FLAGS) -c $< -o $@

$(BUILD)/hip-q8-parity: $(BUILD)/hip-q8-parity.o $(BUILD)/gpu_offload_hip.o
	$(HIPCC) $(HIP_ARCH_FLAGS) $^ -o $@

# GPU-required tests are separate so default 'make check' never depends on ROCm.
hip-check: hip-toolchain-check $(BUILD)/hip-smoke-test $(BUILD)/hip-q8-parity
	$(BUILD)/hip-smoke-test
	$(BUILD)/hip-q8-parity
else
hip-check:
	@echo "LinMoE: hip-check requires BACKEND=hip." >&2
	@exit 2
endif

# Restrict cleanup to the selected backend's reproducible build directory.
clean:
	rm -rf $(BUILD)
