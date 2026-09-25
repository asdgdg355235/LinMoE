#pragma once
/* Small OS boundary for the existing x86 runtime. Allocation pairs remain
 * explicit: aligned buffers use lm_aligned_free; temporary buffers use free.
 * Allocation failure is fatal in this command-line engine, whose callers
 * cannot recover without discarding the current inference state. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#else
#include <time.h>
#endif

static inline void* lm_temp_alloc(size_t bytes) {
    void* p = malloc(bytes ? bytes : 1);
    if (!p) {
        fprintf(stderr, "LinMoE: allocation of %zu bytes failed\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

static inline void* lm_aligned_alloc(size_t bytes, size_t alignment) {
#ifdef _WIN32
    void* p = _aligned_malloc(bytes, alignment);
#else
    void* p = NULL;
    int err = posix_memalign(&p, alignment, bytes ? bytes : 1);
    if (err) errno = err;
#endif
    if (!p) {
        fprintf(stderr, "LinMoE: aligned allocation bytes=%zu alignment=%zu: %s\n",
                bytes, alignment, strerror(errno));
        exit(EXIT_FAILURE);
    }
    return p;
}

static inline void lm_aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* Tick counts are monotonic; use lm_clock_frequency for unit conversion. */
static inline int64_t lm_clock_now(void) {
#ifdef _WIN32
    LARGE_INTEGER value;
    if (!QueryPerformanceCounter(&value)) abort();
    return value.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("LinMoE: clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (int64_t)value.tv_sec * INT64_C(1000000000) + value.tv_nsec;
#endif
}

static inline int64_t lm_clock_frequency(void) {
#ifdef _WIN32
    LARGE_INTEGER value;
    if (!QueryPerformanceFrequency(&value)) abort();
    return value.QuadPart;
#else
    return INT64_C(1000000000);
#endif
}

/* GGUF offsets exceed 4 GiB. POSIX targets must define _FILE_OFFSET_BITS=64
 * and _POSIX_C_SOURCE=200809L before any system header (the build does this). */
#ifdef _WIN32
#define lm_fseek _fseeki64
#define lm_ftell _ftelli64
#else
#define lm_fseek fseeko
#define lm_ftell ftello
#endif
