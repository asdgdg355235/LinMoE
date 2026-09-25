#pragma once
/*
 * Minimal GGUF Parser — extract tensor metadata for expert weight loading
 *
 * We only need: tensor name, type, dimensions, and data offset.
 * No mmap, no full model loading — just read the header to build
 * an offset table for direct I/O reads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include "../platform/runtime.h"

#define MAX_TENSORS 4000
#define MAX_SHARDS 8
#define MAX_NAME_LEN 256
#define MAX_DIMS 4

/* GGML type IDs we care about */
#define GGML_TYPE_F32    0
#define GGML_TYPE_F16    1
#define GGML_TYPE_Q4_0   2
#define GGML_TYPE_Q8_0   8
#define GGML_TYPE_Q4_K  12
#define GGML_TYPE_Q5_K  13
#define GGML_TYPE_Q6_K  14
#define GGML_TYPE_IQ2_XXS 16
#define GGML_TYPE_IQ1_M  22

/* Block sizes per type */
static int ggml_block_size(int type) {
    switch (type) {
        case GGML_TYPE_F32:    return 4;    /* 1 weight per 4 bytes */
        case GGML_TYPE_F16:    return 2;
        case GGML_TYPE_Q4_0:   return 18;   /* 32 weights per block */
        /* Q8_0 stores an FP16 scale followed by 32 signed bytes. */
        case GGML_TYPE_Q8_0:   return 34;
        case GGML_TYPE_Q4_K:   return 144;  /* 256 weights per block */
        case GGML_TYPE_Q5_K:   return 176;
        case GGML_TYPE_Q6_K:   return 210;
        case GGML_TYPE_IQ2_XXS: return 66;
        default: return 0;
    }
}

static int ggml_block_weights(int type) {
    switch (type) {
        case GGML_TYPE_F32:    return 1;
        case GGML_TYPE_F16:    return 1;
        case GGML_TYPE_Q4_0:   return 32;
        case GGML_TYPE_Q8_0:   return 32;
        case GGML_TYPE_Q4_K:   return 256;
        case GGML_TYPE_Q5_K:   return 256;
        case GGML_TYPE_Q6_K:   return 256;
        case GGML_TYPE_IQ2_XXS: return 256;
        default: return 1;
    }
}

typedef struct {
    char name[MAX_NAME_LEN];
    int type;
    int n_dims;
    uint64_t dims[MAX_DIMS];
    uint64_t offset;     /* byte offset in GGUF file (relative to data_start) */
    uint64_t data_size;  /* total bytes of tensor data */
    int shard;           /* which shard file this tensor lives in (0-based) */
} TensorInfo;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    TensorInfo tensors[MAX_TENSORS];
    uint64_t data_start;  /* byte offset where tensor data begins (per shard) */
    int num_shards;
    char shard_paths[MAX_SHARDS][512];
    uint64_t shard_data_starts[MAX_SHARDS];

    /* Model config (extracted from KV pairs) */
    int hidden_dim;
    int expert_intermediate;
    int feed_forward_length;  /* for shared expert (non-MoE FFN) */
    int num_experts;
    int num_layers;
    int expert_used_count;
    int head_count;
    int head_count_kv;
    int ssm_state_size;
    int ssm_inner_size;
    int ssm_conv_kernel;
    int ssm_group_count;  /* num_key_heads for DeltaNet (16 for both 397B and 35B) */
    float rope_theta;
    float routed_scaling_factor;  /* MoE expert output scaling (DeepSeek/Qwen3.5 style) */
    float rms_epsilon;            /* RMSNorm epsilon (we hardcode 1e-6, model might differ) */
} GGUFModel;

/* A bounded reader makes every metadata read/skip fail closed. Endian decoding
 * is explicit; tensor payloads still require the engine's little-endian x86 ABI.
 * Failure is sticky so cleanup stays local to parse_gguf_shard. */
typedef struct {
    FILE* f;
    const char* path;
    uint64_t size, pos;
    int failed;
} GGUFReader;

static void gguf_fail(GGUFReader* r, const char* reason) {
    if (!r->failed)
        fprintf(stderr, "LinMoE: GGUF path=%s offset=%llu: %s\n",
                r->path, (unsigned long long)r->pos, reason);
    r->failed = 1;
}

static void gguf_read(GGUFReader* r, void* out, size_t n) {
    if (r->failed) return;
    if (n > r->size - r->pos || fread(out, 1, n, r->f) != n) {
        gguf_fail(r, "truncated or unreadable metadata");
        return;
    }
    r->pos += n;
}

static uint64_t gguf_uint(GGUFReader* r, size_t n) {
    unsigned char bytes[8] = {0};
    gguf_read(r, bytes, n);
    uint64_t value = 0;
    for (size_t i = 0; i < n; ++i) value |= (uint64_t)bytes[i] << (8 * i);
    return value;
}

static void gguf_skip(GGUFReader* r, uint64_t n) {
    if (r->failed) return;
    if (n > r->size - r->pos || lm_fseek(r->f, (int64_t)n, SEEK_CUR) != 0) {
        gguf_fail(r, "metadata skip exceeds file or seek failed");
        return;
    }
    r->pos += n;
}

/* Required names must fit the engine's fixed lookup keys. String values which
 * are not used by inference are skipped without allocating their contents. */
static void gguf_name(GGUFReader* r, char* out, size_t capacity) {
    uint64_t n = gguf_uint(r, 8);
    out[0] = 0;
    if (n >= capacity) { gguf_fail(r, "metadata/tensor name too long"); return; }
    gguf_read(r, out, (size_t)n);
    if (r->failed) return;
    if (memchr(out, 0, (size_t)n)) gguf_fail(r, "embedded NUL in name");
    out[n] = 0;
}

static void gguf_skip_value(GGUFReader* r, uint32_t type) {
    /* GGUF bool is one byte; u64/i64/f64 are eight. The old table reversed
     * these sizes and could desynchronize the entire tensor directory. */
    static const unsigned sizes[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (type == 8) {
        uint64_t n = gguf_uint(r, 8);
        gguf_skip(r, n);
    } else if (type == 9) {
        uint32_t element = (uint32_t)gguf_uint(r, 4);
        uint64_t count = gguf_uint(r, 8);
        if (element == 8) {
            /* Every string has at least an eight-byte length prefix. */
            if (count > (r->size - r->pos) / 8) {
                gguf_fail(r, "string array exceeds file"); return;
            }
            for (uint64_t i = 0; i < count && !r->failed; ++i) gguf_skip_value(r, 8);
        } else if (element < 13 && sizes[element]) {
            if (count > UINT64_MAX / sizes[element]) gguf_fail(r, "array size overflow");
            else gguf_skip(r, count * sizes[element]);
        } else gguf_fail(r, "unsupported metadata array element type");
    } else if (type < 13 && sizes[type]) gguf_skip(r, sizes[type]);
    else gguf_fail(r, "unknown metadata type");
}

/* Parse one shard into the existing tensor directory. Only the first shard
 * establishes model configuration; each shard has its own alignment/offsets.
 * No partially parsed model may be used after a nonzero return. */
static int parse_gguf_shard(const char* path, GGUFModel* model, int shard) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "LinMoE: open GGUF %s: %s\n", path, strerror(errno));
        return -1;
    }
    int64_t end;
    if (lm_fseek(f, 0, SEEK_END) != 0 || (end = lm_ftell(f)) < 0 ||
        lm_fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "LinMoE: cannot determine GGUF size: %s\n", path);
        fclose(f); return -1;
    }
    GGUFReader r = {f, path, (uint64_t)end, 0, 0};
    char magic[4] = {0};
    gguf_read(&r, magic, 4);
    if (memcmp(magic, "GGUF", 4)) gguf_fail(&r, "invalid little-endian GGUF magic");
    uint32_t version = (uint32_t)gguf_uint(&r, 4);
    uint64_t nt = gguf_uint(&r, 8), nkv = gguf_uint(&r, 8);
    if (version != 2 && version != 3) gguf_fail(&r, "only GGUF v2/v3 are supported");
    if (shard == 0) model->version = version;
    else if (version != model->version) gguf_fail(&r, "shard version mismatch");
    if (nt > MAX_TENSORS - model->n_tensors) gguf_fail(&r, "tensor directory exceeds MAX_TENSORS");
    if (nkv > r.size / 12) gguf_fail(&r, "metadata count exceeds file");
    uint32_t alignment = 32;
    for (uint64_t kv = 0; kv < nkv && !r.failed; ++kv) {
        char key[512];
        gguf_name(&r, key, sizeof(key));
        uint32_t type = (uint32_t)gguf_uint(&r, 4);
        if (!strcmp(key, "general.alignment") && type != 4)
            gguf_fail(&r, "general.alignment must be uint32");
        if (type == 4) {
            uint32_t val = (uint32_t)gguf_uint(&r, 4);
            if (!strcmp(key, "general.alignment")) alignment = val;
            if (shard != 0) continue;
            /* All runtime dimensions use signed int. Refuse truncation before
             * any products, division or allocations use model metadata. */
            int* field = NULL;
            if (strstr(key, "expert_count")) field = &model->num_experts;
            if (strstr(key, "expert_used_count")) field = &model->expert_used_count;
            if (strstr(key, "expert_feed_forward_length")) field = &model->expert_intermediate;
            else if (strstr(key, "feed_forward_length")) field = &model->feed_forward_length;
            if (strstr(key, "embedding_length")) field = &model->hidden_dim;
            if (strstr(key, "block_count")) field = &model->num_layers;
            if (strstr(key, "head_count_kv")) field = &model->head_count_kv;
            else if (strstr(key, "head_count")) field = &model->head_count;
            if (strstr(key, "ssm.state_size")) field = &model->ssm_state_size;
            if (strstr(key, "ssm.inner_size")) field = &model->ssm_inner_size;
            if (strstr(key, "ssm.conv_kernel")) field = &model->ssm_conv_kernel;
            if (strstr(key, "ssm.group_count")) field = &model->ssm_group_count;
            if (field) {
                if (val > INT_MAX) gguf_fail(&r, "model dimension exceeds INT_MAX");
                else *field = (int)val;
            }
        } else if (type == 6) {
            uint32_t bits = (uint32_t)gguf_uint(&r, 4);
            float val; memcpy(&val, &bits, sizeof(val));
            if (shard != 0) continue;
            if (strstr(key, "rope.freq_base")) model->rope_theta = val;
            if (strstr(key, "routed_scaling_factor")) model->routed_scaling_factor = val;
            if (strstr(key, "rms_epsilon")) model->rms_epsilon = val;
        } else {
            /* Includes the entire dimension_sections array, even past 16
             * entries. Logging a prefix must never leave the cursor mid-value. */
            gguf_skip_value(&r, type);
        }
    }
    if (!alignment || (alignment & (alignment - 1))) gguf_fail(&r, "invalid GGUF alignment");
    uint64_t first = model->n_tensors;
    for (uint64_t ti = 0; ti < nt && !r.failed; ++ti) {
        TensorInfo* t = &model->tensors[model->n_tensors];
        gguf_name(&r, t->name, sizeof(t->name));
        uint32_t nd = (uint32_t)gguf_uint(&r, 4);
        if (nd < 1 || nd > MAX_DIMS) { gguf_fail(&r, "invalid tensor rank"); break; }
        t->n_dims = (int)nd;
        uint64_t elements = 1;
        for (uint32_t d = 0; d < nd && !r.failed; ++d) {
            t->dims[d] = gguf_uint(&r, 8);
            if (!t->dims[d] || t->dims[d] > UINT64_MAX / elements)
                gguf_fail(&r, "invalid/overflowing tensor dimensions");
            else elements *= t->dims[d];
        }
        uint32_t type = (uint32_t)gguf_uint(&r, 4);
        t->offset = gguf_uint(&r, 8);
        if (type > INT_MAX) { gguf_fail(&r, "invalid tensor type"); break; }
        t->type = (int)type;
        t->shard = shard;
        int bw = ggml_block_weights(t->type), bs = ggml_block_size(t->type);
        if (!bs || t->dims[0] % (unsigned)bw || elements / bw > UINT64_MAX / (unsigned)bs) {
            gguf_fail(&r, "unsupported type or invalid quantized tensor size"); break;
        }
        t->data_size = (elements / bw) * bs;
        /* Names are unique across shards; a duplicate makes lookup ambiguous. */
        for (uint64_t prior = 0; prior < model->n_tensors; ++prior) {
            if (!strcmp(model->tensors[prior].name, t->name)) {
                gguf_fail(&r, "duplicate tensor name"); break;
            }
        }
        if (!r.failed) ++model->n_tensors;
    }
    if (!r.failed) {
        uint64_t start = (r.pos + alignment - 1) & ~((uint64_t)alignment - 1);
        if (start > r.size) gguf_fail(&r, "tensor data start exceeds file");
        for (uint64_t ti = first; ti < model->n_tensors && !r.failed; ++ti) {
            TensorInfo* t = &model->tensors[ti];
            if (t->offset % alignment || t->offset > r.size - start ||
                t->data_size > r.size - start - t->offset)
                gguf_fail(&r, "tensor range/alignment invalid");
        }
        model->shard_data_starts[shard] = start;
        if (shard == 0) model->data_start = start;
    }
    fclose(f);
    return r.failed ? -1 : 0;
}

/* Entry point retained for standalone single-file reference programs. */
static int parse_gguf(const char* path, GGUFModel* model) {
    memset(model, 0, sizeof(*model));
    return parse_gguf_shard(path, model, 0);
}

/* Split files must be supplied by their first-shard filename. Every advertised
 * shard must exist and validate; missing shards are never silently skipped. */
static int parse_gguf_split(const char* path, GGUFModel* model) {
    memset(model, 0, sizeof(*model));
    if (strlen(path) >= sizeof(model->shard_paths[0])) {
        fprintf(stderr, "LinMoE: GGUF path exceeds runtime path limit\n"); return -1;
    }
    const char* name = strrchr(path, '/');
    const char* backslash = strrchr(path, '\\');
    if (!name || (backslash && backslash > name)) name = backslash;
    name = name ? name + 1 : path;
    size_t len = strlen(name);
    const char* pattern = NULL;
    unsigned total = 1, first = 1;
    /* Match only the filename suffix, never a directory containing -of-. */
    if (len >= 20) {
        const char* suffix = name + len - 20;
        int consumed = 0;
        if (sscanf(suffix, "-%5u-of-%5u.gguf%n", &first, &total, &consumed) == 2 && consumed == 20)
            pattern = suffix + 1;
        else total = first = 1;
    }
    if (first != 1 || total < 1 || total > MAX_SHARDS) {
        fprintf(stderr, "LinMoE: require first GGUF shard, total <= %d: %s\n", MAX_SHARDS, path);
        return -1;
    }
    model->num_shards = (int)total;
    for (unsigned i = 0; i < total; ++i) {
        char* dest = model->shard_paths[i];
        strcpy(dest, path);
        if (pattern) {
            char number[16];
            snprintf(number, sizeof(number), "%05u", i + 1);
            memcpy(dest + (pattern - path), number, 5);
        }
        if (parse_gguf_shard(dest, model, (int)i)) return -1;
    }
    return 0;
}

/* Tensor pointers remain valid for the lifetime of the model directory. */
static TensorInfo* find_tensor(GGUFModel* model, const char* name) {
    for (uint64_t i = 0; i < model->n_tensors && i < MAX_TENSORS; ++i)
        if (!strcmp(model->tensors[i].name, name)) return &model->tensors[i];
    return NULL;
}
