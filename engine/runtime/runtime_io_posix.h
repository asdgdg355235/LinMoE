#pragma once
/* Native Linux correctness baseline. Start/wait preserves the scheduler's
 * interface but start completes synchronously: this path claims no overlap.
 * A failed read terminates the CLI before any consumer can use partial data. */
#include "../platform/posix_io.h"
#include <limits.h>

static LmFile g_shard_handles[MAX_SHARDS];

static void open_shard_handles(GGUFModel* model) {
    for (int i = 0; i < model->num_shards; ++i) {
        if (lm_file_open(&g_shard_handles[i], model->shard_paths[i]) != 0) {
            for (int j = 0; j < i; ++j) lm_file_close(&g_shard_handles[j]);
            exit(EXIT_FAILURE);
        }
    }
    fprintf(stderr, "LinMoE storage: buffered pread, synchronous, %d shards\n",
            model->num_shards);
}

static void close_shard_handles(GGUFModel* model) {
    for (int i = 0; i < model->num_shards; ++i)
        lm_file_close(&g_shard_handles[i]);
}

/* Return ordinary malloc storage, matching all existing tensor owners. */
static void* read_tensor_from_handle(LmFile file, uint64_t data_start,
                                    TensorInfo* tensor) {
    if (tensor->offset > UINT64_MAX - data_start ||
        tensor->data_size > SIZE_MAX || tensor->data_size > INT_MAX) {
        fprintf(stderr, "LinMoE: tensor %s exceeds runtime size/offset limits\n", tensor->name);
        exit(EXIT_FAILURE);
    }
    void* result = lm_temp_alloc((size_t)tensor->data_size);
    if (lm_file_read(&file, data_start + tensor->offset, result,
                     (size_t)tensor->data_size) != 0) {
        fprintf(stderr, "LinMoE: loading tensor %s failed\n", tensor->name);
        free(result);
        exit(EXIT_FAILURE);
    }
    return result;
}

/* Some callers address staging + offset%ALIGN. Preserve that layout, but
 * read only the payload: buffered pread does not require enclosing reads or
 * padding beyond EOF. No consumer may access the unused staging prefix. */
static int read_bytes_from_shard(int shard, uint64_t offset, void* dest,
                                int size, void* staging, int capacity) {
    size_t sub = (size_t)(offset % ALIGN);
    if (size < 0 || shard < 0 || shard >= MAX_SHARDS ||
        (!dest && (!staging || capacity < 0 || sub > (size_t)capacity ||
                   (size_t)size > (size_t)capacity - sub))) {
        fprintf(stderr, "LinMoE: invalid read shard=%d offset=%llu size=%d capacity=%d\n",
                shard, (unsigned long long)offset, size, capacity);
        exit(EXIT_FAILURE);
    }
    void* target = dest ? dest : (char*)staging + sub;
    if (lm_file_read(&g_shard_handles[shard], offset, target, (size_t)size) != 0)
        exit(EXIT_FAILURE);
    return size;
}

typedef struct { int valid; } AsyncRead;

static void async_read_start(AsyncRead* ar, int shard, uint64_t offset,
                             void* dest, int size, void* staging, int capacity) {
    ar->valid = 0;
    read_bytes_from_shard(shard, offset, dest, size, staging, capacity);
    ar->valid = 1;
}

static void async_read_wait(AsyncRead* ar) {
    /* No in-flight work exists in this baseline. Buffers are immediately
     * reusable after start; a future async backend must change this contract. */
    ar->valid = 0;
}
