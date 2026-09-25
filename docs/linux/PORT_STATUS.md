# LinMoE native Linux foundation

Base audited: `asdgdg355235/LinMoE`, `master`,
`6aef92d7a947566b650b3231aa3169d71b26f15e`.
Target direction: Arch Linux, RX 6950 XT / gfx1030, HIP, NVMe-streamed MoE.
This change implements a **CPU foundation**, not the completed AMD port.

## What builds and runs

- Native Linux C11/C++17/OpenMP build with GCC; no Wine, CUDA or HIP dependency.
- CPU executable `build/linux-cpu/linmoe` and metadata tool `linmoe-inspect`.
- Checked buffered `pread` storage: explicit offsets, 64-bit addressing, retry on
  EINTR, short-read completion, EOF/range checks and diagnostic failures.
- Existing scheduler start/wait interface backed by synchronous reads on Linux.
  There is **no Linux asynchronous overlap** in this milestone.
- Synthetic two-layer GQA/DeltaNet inference, two forward positions, six cold
  expert loads followed by six cache hits, and an analytically known logit row.
- Existing CUDA source and extracted Win32 I/O retained as reference, untested.

**CPU requirement:** AVX512F/BW/DQ/VL, AVX2, FMA and F16C. Several inherited
float kernels unconditionally use AVX-512. This is a host CPU requirement, not
an AMD GPU requirement. Do not infer support for an AVX2-only CPU from x86-64.
No AVX2-only inference build is claimed. The metadata inspector needs no SIMD.

```sh
make -j2
make check
./build/linux-cpu/linmoe-inspect /absolute/path/to/model-00001-of-00006.gguf
```

For a future real-model comparison (not performed here), use the exact same
model, token IDs, K, and floating-point configuration as the reference:

```sh
WINMOE_TOPK=10 WINMOE_CACHE_GB=1 OMP_NUM_THREADS=2 \
WINMOE_DUMP_ALL_LOGITS=linux-logits.bin WINMOE_TRACE_OUT=linux-trace.tsv \
./build/linux-cpu/linmoe --model /absolute/path/to/model-00001-of-00006.gguf \
  --prompt-tokens 248045,846,198,9419,248046,198,248045,74455,198 --tokens 10
```

That example is for the inherited Qwen vocabulary and K=10 configuration.
`--tokens` counts total forward positions, **including prompt positions**;
there is no text tokenizer in the C entry point. RAM for common weights,
state, and staging is additional to `WINMOE_CACHE_GB`. The inherited default
cache budget is 20 GiB. K is still capped at 8 by default; the explicit override
above restores K=10 for comparison. `WINMOE_*` names remain compatible.

## Evidence

Test environment: Linux x86-64 container, GCC 13, host reports AVX-512 features.
No AMD device, ROCm compiler/runtime, real model, or target Arch machine was
available. No throughput improvement or real-model equivalence is claimed.

| Check | Result |
|---|---|
| `make -j2 check` | Pass |
| GGUF synthetic cases | 133 pass, including every incomplete prefix of a valid fixture |
| Q6_K scalar versus AVX2 | 3,000 pairs pass; max abs 3.662109e-4, RMSE 4.061567e-5 in the release build |
| Q8 signed integer dot | All 65,536 constant int8 pairs plus mixed lanes pass exactly |
| POSIX reads | Unaligned payloads, exact EOF, sparse >4 GiB, post-open truncation, invalid ranges/fds pass |
| Analytic inference | GQA + DeltaNet, cold/hot cache, all 64 emitted logits checked |
| ASan + UBSan | Entire `make check` suite passes with leak detection disabled |
| LeakSanitizer | Environment rejects its tracing/thread inspection; no leak-free claim |
| Real Qwen CPU/CUDA/HIP comparison | Not run |
| RX 6950 XT execution | Not run |
| Windows/CUDA regression | Not run |

Sanitizer reproduction:

```sh
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
make -j2 BUILD=build/sanitize \
  CFLAGS='-O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  CXXFLAGS='-O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer' check
```

Disable leak detection only where the host has the same LeakSanitizer restriction.
The analytic model has zero attention/expert weights: it verifies integration,
allocation, storage, cache, and an expected output, **not general kernel parity**.

## Repository map from current source

| Concern | Implementation / finding |
|---|---|
| Actual inference entry | `engine/runtime/winmoe_inference.c`: model load, token loop, expert scheduling, RAM cache, profiling, trace/logit dumps |
| Metadata and shards | `gguf_parser.h`: fixed 4,000 tensor / 8 shard directory; bounded parsing and range validation in this change |
| CPU math | `attention.h`, `transformer.h`, `deltanet_impl.h`, `q4k_dequant.h`, `q5k_dequant.h`, `q6k_dequant.h`, `q8_dequant.h`, `q8k_quant.h` |
| IQ2 reference | `iq2_dequant*.h` and legacy expert helpers; not a supported generic matvec dispatch in the active engine |
| GPU boundary | `gpu_offload.h`; CPU implementation declines initialization and exposes no GPU data |
| CUDA implementation | `gpu_offload.cu`: Q8/Q4/Q5 matvecs, expert/SwiGLU/fused/batch paths, FP32 router, weight/cache allocation, streams/events |
| GPU attention split | GQA/DeltaNet projections offloaded; attention and recurrence remain in host code |
| CUDA synchronization | Compute/H2D/D2H/expert streams; two pipeline slots with pinned result buffers and event dependencies; synchronous expert paths also present |
| cuBLAS | Handle creation/destruction present; no BLAS inference calls found, so no hipBLAS dependency justified yet |
| Active RAM cache | Arrays inside `main`, 20 GiB default, round-robin or opt-in LFU; not the separate cache class |
| GPU cache limits | 300-entry compile-time maximum, default 200 requested from main; no dynamic memory-budget implementation |
| Native storage | `engine/platform/posix_io.*` and `runtime_io_posix.h`, buffered synchronous baseline |
| Win32 storage | `runtime_io_win32.h`, extracted legacy unbuffered/OVERLAPPED implementation |
| Allocation/timing | `engine/platform/runtime.h`, paired aligned allocation and monotonic ticks; heap temporaries with explicit frees |
| Other engine files | `engine.cpp`, scheduler/cache classes, slab/replay/integrated benchmarks are historical or experimental paths; not linked into native inference |
| Python tools | Repacking, trace analysis, model/reference comparisons and experiments remain separate; many scripts contain Windows paths |
| Build | New root Makefile; existing manual MSVC/nvcc instructions retained as historical reference |
| Portability backlog | Win32 APIs remain in legacy benchmarks and tests; no io_uring, O_DIRECT or HIP implementation added |

## Defects addressed for this baseline

1. Q8_0 metadata sized as FP32: corrected to 34 bytes per 32 values, eliminating
   oversized reads and allocations. Unknown encodings no longer assume FP32.
2. Wrong bool/u64 metadata widths, truncated reads, excessive ranks/counts,
   overflowing dimensions, ignored alignment and silently missing shards:
   the common bounded shard parser rejects these conditions.
3. Platform dependence: native read/clock/alignment implementation and a real
   CPU-only link target. Backend requests other than `cpu` fail at build time.
4. Q8 VNNI sign adjustment mishandled activation -128; AVX2 widening performs
   exact signed products and removes the unconditional VNNI requirement.
5. GQA reused a query buffer sized from DeltaNet. The synthetic fixture failed
   with a heap overrun; ASan located it in query deinterleaving. Allocation now
   uses actual GQA projection maxima and validates head geometry.
6. Out-of-vocabulary debug probes and invalid prompt indices: checked before
   indexing. Unused diagnostic expert IDs are initialized.
7. Temporary recurrence allocations and process-owned weights/cache/state:
   matching frees added, with GPU shutdown ordered before host release.
8. Failed expert allocation formerly fed an unrelated scratch buffer to math:
   now fatal. Unsupported matvec encodings and embeddings fail diagnostically.

## Open correctness gates before HIP/full-model claims

- **CPU/CUDA DeltaNet differ:** CPU `deltanet_forward` maps a head with
  `h / DN_HEADS_PER_GROUP`; the GPU-assisted host recurrence maps with
  `h_idx % DN_NUM_KV_GROUPS`. They cannot both be used interchangeably as an
  oracle when there are multiple groups. Resolve against intermediate tensors
  from a pinned known-good reference; the one-group smoke fixture cannot do so.
- **Kernel parity is incomplete:** Q4/Q5 float fallbacks and packed integer
  paths need independent block fixtures; quantized activation error requires
  separate accounting. FP32, Q8, SwiGLU, GQA and recurrence need nonzero tests.
- **Model schema validation is incomplete:** tensor file bounds are checked,
  but every model-specific shape/type relationship is not yet validated.
  Do not treat this as a hardened executor for arbitrary untrusted GGUFs.
- Some inherited allocations, trace writes and numerical failure paths still
  lack complete error propagation. GPU return codes and initialization cleanup
  need systematic review before enabling HIP.
- Legacy Win32 read completion errors and synchronous-handle async fallback
  remain unsafe in the reference path; the native reader does not use them.
- CUDA has wave32 masks/reductions, fixed dimensions and launch geometries,
  fixed pipeline allocations, unchecked calls and incomplete shutdown coverage.
  HIP translation must review these, not just rename APIs.
- Expert cache entries and read pools assume uniform per-layer expert byte
  sizes. This baseline explicitly rejects mismatches; generalized sizing and
  request ownership belong in the subsequent streaming milestone.

## Next implementation order

1. Resolve the DeltaNet mapping discrepancy and establish nonzero CPU operation
   fixtures plus real-model intermediate/logit reference comparisons.
2. Decide an AVX2-only host baseline, preserving existing fast paths behind
   explicit build/dispatch choices rather than assuming AVX-512 availability.
3. Add an optional HIP initialization/allocation/copy test with gfx1030 build
   discovery, clear runtime diagnostics, and device execution on the RX 6950 XT.
4. Port and independently compare Q8 operations, then attention/projections and
   Q4/Q5/expert operations. Preserve CPU subsystem switches.
5. Measure the buffered storage baseline, then add O_DIRECT alignment discovery
   and io_uring incrementally, with cancellation and buffer lifetime tests.
6. Query VRAM budgets with headroom, instrument residency and stalls, and measure
   disk/H2D/compute overlap before optimizing it.

Primary format/API references used for this baseline:
- https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- https://www.man7.org/linux/man-pages/man2/pread.2.html

Historical Windows/NVIDIA benchmark numbers elsewhere in this repository are
not LinMoE Linux/AMD measurements.
