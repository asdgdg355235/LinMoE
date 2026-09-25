/* CPU-only backend for the existing GPU interface. Initialization explicitly
 * declines offload, so inference follows its CPU paths. Query functions return
 * empty state; compute/upload functions return failure without touching output.
 * Keep signatures identical to gpu_offload.h to catch interface drift. */
#include "gpu_offload.h"
#include <stdio.h>
#include <stddef.h>

int gpu_init(void) {
    fprintf(stderr, "LinMoE backend: CPU (no GPU backend compiled)\n");
    return -1;
}

void gpu_shutdown(void) {
}

int gpu_is_initialized(void) {
    return 0;
}

/* CPU has no accelerator implementation behind the high-level GPU API. */
int gpu_supports_full_inference(void) {
    return 0;
}

float gpu_vram_used_mb(void) {
    return 0.0f;
}

int gpu_upload_deltanet_weights(int layer,
    const void* qkv_q8, int qkv_rows, int qkv_cols,
    const void* gate_q8, int gate_rows, int gate_cols,
    const void* ssm_out_q8, int ssm_rows, int ssm_cols) {
    (void)layer;
    (void)qkv_q8;
    (void)qkv_rows;
    (void)qkv_cols;
    (void)gate_q8;
    (void)gate_rows;
    (void)gate_cols;
    (void)ssm_out_q8;
    (void)ssm_rows;
    (void)ssm_cols;
    return -1;
}

int gpu_deltanet_projections(int layer,
    const float* normed, int hidden_dim,
    float* qkv_out, int qkv_dim,
    float* gate_out, int gate_dim) {
    (void)layer;
    (void)normed;
    (void)hidden_dim;
    (void)qkv_out;
    (void)qkv_dim;
    (void)gate_out;
    (void)gate_dim;
    return -1;
}

int gpu_ssm_out_projection(int layer,
    const float* gated, int gated_dim,
    float* output, int output_dim) {
    (void)layer;
    (void)gated;
    (void)gated_dim;
    (void)output;
    (void)output_dim;
    return -1;
}

int gpu_launch_qkv_gate(int layer, int slot,
    const float* normed, int hidden_dim,
    int qkv_dim, int gate_dim) {
    (void)layer;
    (void)slot;
    (void)normed;
    (void)hidden_dim;
    (void)qkv_dim;
    (void)gate_dim;
    return -1;
}

int gpu_wait_qkv_gate(int slot) {
    (void)slot;
    return -1;
}

int gpu_launch_ssm_out(int layer, int slot,
    const float* gated, int gated_dim, int output_dim) {
    (void)layer;
    (void)slot;
    (void)gated;
    (void)gated_dim;
    (void)output_dim;
    return -1;
}

int gpu_wait_ssm_out(int slot) {
    (void)slot;
    return -1;
}

float* gpu_get_qkv_out(int slot) {
    (void)slot;
    return NULL;
}

float* gpu_get_gate_out(int slot) {
    (void)slot;
    return NULL;
}

float* gpu_get_ssm_out_buf(int slot) {
    (void)slot;
    return NULL;
}

int gpu_upload_gqa_weights(int layer,
    const void* wq_q8, int wq_rows, int wq_cols,
    const void* wk_q8, int wk_rows, int wk_cols,
    const void* wv_q8, int wv_rows, int wv_cols,
    const void* wo_q8, int wo_rows, int wo_cols) {
    (void)layer;
    (void)wq_q8;
    (void)wq_rows;
    (void)wq_cols;
    (void)wk_q8;
    (void)wk_rows;
    (void)wk_cols;
    (void)wv_q8;
    (void)wv_rows;
    (void)wv_cols;
    (void)wo_q8;
    (void)wo_rows;
    (void)wo_cols;
    return -1;
}

int gpu_gqa_projections(int layer,
    const float* normed, int hidden_dim,
    float* q_gate_out, int q_gate_dim,
    float* k_out, int k_dim, float* v_out, int v_dim) {
    (void)layer;
    (void)normed;
    (void)hidden_dim;
    (void)q_gate_out;
    (void)q_gate_dim;
    (void)k_out;
    (void)k_dim;
    (void)v_out;
    (void)v_dim;
    return -1;
}

int gpu_gqa_output(int layer,
    const float* attn_out, int attn_dim,
    float* output, int hidden_dim) {
    (void)layer;
    (void)attn_out;
    (void)attn_dim;
    (void)output;
    (void)hidden_dim;
    return -1;
}

int gpu_upload_router(int layer, const float* weights, int hidden_dim, int num_experts) {
    (void)layer;
    (void)weights;
    (void)hidden_dim;
    (void)num_experts;
    return -1;
}

int gpu_router(int layer, const float* normed, int hidden_dim, float* logits_out, int num_experts) {
    (void)layer;
    (void)normed;
    (void)hidden_dim;
    (void)logits_out;
    (void)num_experts;
    return -1;
}

int gpu_cache_expert(int layer, int expert_id,
    const void* gate_data, int gate_size,
    const void* up_data, int up_size,
    const void* down_data, int down_size) {
    (void)layer;
    (void)expert_id;
    (void)gate_data;
    (void)gate_size;
    (void)up_data;
    (void)up_size;
    (void)down_data;
    (void)down_size;
    return -1;
}

int gpu_find_cached_expert(int layer, int expert_id) {
    (void)layer;
    (void)expert_id;
    return -1;
}

int gpu_expert_ffn(int cache_idx, const float* input, int hidden_dim,
    int intermediate, float* gate_out, float* up_out, float* expert_out,
    int gate_type, int down_type) {
    (void)cache_idx;
    (void)input;
    (void)hidden_dim;
    (void)intermediate;
    (void)gate_out;
    (void)up_out;
    (void)expert_out;
    (void)gate_type;
    (void)down_type;
    return -1;
}

int gpu_expert_down(int cache_idx, const float* act, int intermediate,
    float* output, int hidden_dim) {
    (void)cache_idx;
    (void)act;
    (void)intermediate;
    (void)output;
    (void)hidden_dim;
    return -1;
}

int gpu_expert_ffn_fused(int cache_idx, const float* input, int hidden_dim,
    int intermediate, float* expert_out) {
    (void)cache_idx;
    (void)input;
    (void)hidden_dim;
    (void)intermediate;
    (void)expert_out;
    return -1;
}

int gpu_expert_batch_start(const float* normed, int hidden_dim) {
    (void)normed;
    (void)hidden_dim;
    return -1;
}

int gpu_expert_batch_add(int cache_idx, int hidden_dim, int intermediate, float weight) {
    (void)cache_idx;
    (void)hidden_dim;
    (void)intermediate;
    (void)weight;
    return -1;
}

int gpu_expert_batch_finish(float* moe_out, int hidden_dim) {
    (void)moe_out;
    (void)hidden_dim;
    return -1;
}

int gpu_expert_cache_count(void) {
    return 0;
}

void gpu_set_expert_limit(int limit) {
    (void)limit;
}

/* Q/K/V capability is independent of the full legacy inference pipeline. */
int gpu_supports_gqa_projections(void) { return 0; }
