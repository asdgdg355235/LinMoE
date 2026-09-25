#pragma once
/*
 * GPU Offload API — Phase 3 Pipeline Architecture
 *
 * Sync API (backward compatible):
 *   gpu_deltanet_projections() — blocking QKV+Gate
 *   gpu_ssm_out_projection() — blocking SSM Out
 *
 * Async API (Phase 3 pipeline):
 *   gpu_launch_qkv_gate() → gpu_wait_qkv_gate() — non-blocking launch, explicit wait
 *   gpu_launch_ssm_out() → gpu_wait_ssm_out() — same pattern
 *   gpu_get_qkv_out() / gpu_get_gate_out() — access pinned result buffers
 */

#ifdef __cplusplus
extern "C" {
#endif

int gpu_init(void);
void gpu_shutdown(void);
int gpu_is_initialized(void);
/* Full inference requires every high-level operation below to be implemented.
 * A backend may initialize successfully for diagnostics/kernel validation while
 * deliberately advertising foundation-only capability. */
int gpu_supports_full_inference(void);
float gpu_vram_used_mb(void);

/* Upload Q8_0 DeltaNet weights to GPU */
int gpu_upload_deltanet_weights(int layer,
    const void* qkv_q8, int qkv_rows, int qkv_cols,
    const void* gate_q8, int gate_rows, int gate_cols,
    const void* ssm_out_q8, int ssm_rows, int ssm_cols);

/* Sync API (blocking) */
int gpu_deltanet_projections(int layer,
    const float* normed, int hidden_dim,
    float* qkv_out, int qkv_dim,
    float* gate_out, int gate_dim);

int gpu_ssm_out_projection(int layer,
    const float* gated, int gated_dim,
    float* output, int output_dim);

/* Async API (non-blocking) — Phase 3 pipeline */
int gpu_launch_qkv_gate(int layer, int slot,
    const float* normed, int hidden_dim,
    int qkv_dim, int gate_dim);
int gpu_wait_qkv_gate(int slot);

int gpu_launch_ssm_out(int layer, int slot,
    const float* gated, int gated_dim, int output_dim);
int gpu_wait_ssm_out(int slot);

/* Access pinned result buffers after wait */
float* gpu_get_qkv_out(int slot);
float* gpu_get_gate_out(int slot);
float* gpu_get_ssm_out_buf(int slot);

/* GPU Standard Attention (GQA) */
int gpu_upload_gqa_weights(int layer,
    const void* wq_q8, int wq_rows, int wq_cols,
    const void* wk_q8, int wk_rows, int wk_cols,
    const void* wv_q8, int wv_rows, int wv_cols,
    const void* wo_q8, int wo_rows, int wo_cols);
int gpu_gqa_projections(int layer,
    const float* normed, int hidden_dim,
    float* q_gate_out, int q_gate_dim,
    float* k_out, int k_dim, float* v_out, int v_dim);
int gpu_gqa_output(int layer,
    const float* attn_out, int attn_dim,
    float* output, int hidden_dim);

/* GPU Router */
int gpu_upload_router(int layer, const float* weights, int hidden_dim, int num_experts);
int gpu_router(int layer, const float* normed, int hidden_dim, float* logits_out, int num_experts);

/* GPU Expert Cache — hot experts computed on GPU at 256 GB/s */
int gpu_cache_expert(int layer, int expert_id,
    const void* gate_data, int gate_size,
    const void* up_data, int up_size,
    const void* down_data, int down_size);
int gpu_find_cached_expert(int layer, int expert_id);
int gpu_expert_ffn(int cache_idx, const float* input, int hidden_dim,
    int intermediate, float* gate_out, float* up_out, float* expert_out,
    int gate_type, int down_type);
int gpu_expert_down(int cache_idx, const float* act, int intermediate,
    float* output, int hidden_dim);
/* Fully fused: gate+up+swiglu+down all on GPU, single upload/download */
int gpu_expert_ffn_fused(int cache_idx, const float* input, int hidden_dim,
    int intermediate, float* expert_out);

/* Batched expert API — single upload/download per layer */
int gpu_expert_batch_start(const float* normed, int hidden_dim);
int gpu_expert_batch_add(int cache_idx, int hidden_dim, int intermediate, float weight);
int gpu_expert_batch_finish(float* moe_out, int hidden_dim);

int gpu_expert_cache_count(void);
/* Configure the maximum number of GPU-cached experts.
 * Round-robin eviction takes over once the cache is full. */
void gpu_set_expert_limit(int limit);

#ifdef __cplusplus
}
#endif
