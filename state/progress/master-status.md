# LinMoE native port status

Current native milestone: CPU executable, bounded GGUF loader and checked
synchronous buffered reads. Synthetic integration/reference checks pass.
HIP and full-model parity remain pending. See
[the current Linux port status](../../docs/linux/PORT_STATUS.md).

The following section is inherited Windows/NVIDIA evidence and is retained
for comparison; it is not a Linux/AMD performance claim.

---

# Master Status — WinMoE Custom Engine

## Current Performance
**6.21 tok/s peak, 5.88 tok/s steady** on Qwen3.5-397B Q4_K_M (228 GB, 6 GGUF shards)
**124x speedup** from first token (0.05 tok/s at v5.0)

## Architecture
```
SSD (228 GB Q4_K_M) → RAM (20 GB expert cache, 73% hit) → GPU (fused expert FFN, 240 cached)
                                                          → GPU (Q8_0 DeltaNet projections, 6.1 GB VRAM)
                                                          → CPU (cold expert FFN, AVX-512+VNNI)
                                                          → CPU (FP32 router, 16ms)
```

## Build Command
```bash
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin:$PATH"
cd engine/runtime

# Compile (need vcvars or MSVC in PATH):
nvcc -O2 -arch=sm_86 -c gpu_offload.cu -o gpu_offload.obj
cl /O2 /arch:AVX512 /openmp /fp:fast /c winmoe_inference.c /Fo:winmoe.obj
link winmoe.obj gpu_offload.obj /OUT:winmoe.exe /STACK:8388608 cublas.lib cuda.lib cudart.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64"

# Run:
OMP_NUM_THREADS=10 ./winmoe.exe --model "D:/models/qwen35-397b-q4/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf" --tokens 30
```

## Profiled Breakdown (warm token, 161-170ms)
- DeltaNet attention: 87-98ms (reported, includes some expert spillover)
- Expert FFN (warm, fused GPU, 32-thread kernels): 38-58ms
- Router (CPU FP32): 25ms
- Other: ~2ms

## Version History
| Version | tok/s | Speedup | Change |
|---------|-------|---------|--------|
| v5.0 | 0.05 | 1x | First 397B token |
| v5.7 | 0.67 | 13x | Buffer bug fix |
| v6.3 | 0.91 | 18x | AVX-512 |
| v6.5 | 1.39 | 28x | Integer accumulation + fusion |
| v7.0 | 1.53 | 31x | VNNI |
| v7.4 | 2.60 | 52x | GPU Q8_0 custom CUDA kernel |
| v7.5 | 2.78 | 56x | Phase 3 async pipeline |
| v8.0 | 2.87 | 57x | Fused gate+up + AVX2 SwiGLU |
| v8.5 | 3.31 | 66x | GPU expert cache (150 limit) |
| v8.9 | 4.37 | 87x | GPU shared memory input caching |
| v9.4 | 4.47 | 89x | Fused GPU expert FFN + Q5_K shmem fix |
| v9.6 | 6.21 | 124x | Expert-optimized 32-thread GPU kernels |

## Exhausted Optimizations
- VNNI for Q4_K/Q5_K (v7.6) — hsum negates gain
- AVX-512 widening Q4_K (v7.7) — nibble mismatch
- Multi-row blocking (v7.8) — already in L1
- Software prefetch (v7.9) — HW handles it
- GPU router (v8.7, v9.5) — VRAM displacement > router savings
- Expert stream separation (v9.1, v9.2) — SM contention
- Expert cache 250 (v9.3) — VRAM pressure

## Next Actions
1. ~~GPU Q4_K/Q5_K kernel optimization~~ — DONE (v9.6, +39%)
2. **Keep pushing toward 7 tok/s** — vectorized loads, loop unrolling, huge pages
3. GQA attention for standard layers (15 layers output zeros)
3. LM head on GPU for real text output
4. Huge pages for expert RAM cache
5. LRU expert cache eviction

## Key Files
- engine/runtime/winmoe_inference.c — main loop
- engine/runtime/gpu_offload.cu — CUDA kernels (Q8_0, Q4_K, Q5_K, SwiGLU, router)
- engine/runtime/results.tsv — 60 experiments
- STUDY.md — full research narrative (2200+ lines)
- docs/paper_draft.md — research paper draft
