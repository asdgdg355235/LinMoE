# LinMoE native Linux foundation

Base audited: `asdgdg355235/LinMoE`, `master`,
`6aef92d7a947566b650b3231aa3169d71b26f15e`.
Target direction: Arch Linux, RX 6950 XT / gfx1030, HIP, NVMe-streamed MoE.
The CPU and HIP/Q8 foundations are established. The complete staged GQA
projection path (Q/K/V + Wo) is now validated on RX 6950 XT / gfx1030; this is
still not a completed AMD inference port.

## GQA Q/K/V/Wo milestone — target validated

Persistent Wq/Wk/Wv/Wo uploads, checked packed sizes, shared Q8 dispatch,
synchronized host publication, four-matrix transactional replacement,
failure-injection rollback, scratch resize/reuse and shutdown cleanup all pass.
Attention itself remains on CPU. Hybrid inference is opt-in with
`WINMOE_GQA_HIP=1`; the full-inference gate remains closed.

Target results:
- Q+gate: max abs 0.001953125, RMSE 0.000416676675.
- K: max abs 0.00161743164, RMSE 0.000359322842.
- V: max abs 0.00146484375, RMSE 0.000346418818.
- Wo: max abs 0.0029296875, RMSE 0.000574647851.
- Hybrid 3-position logits: max abs 1.66893005e-06, RMSE 1.77150513e-07.
- Four-matrix representative residency: 106.25 MiB per GQA layer.

The deliberate invalid-request and out-of-memory diagnostics are negative-test
coverage, not unexpected failures or real device exhaustion. See
[the implementation/validation handoff](HIP_GQA.md) for full details.

## HIP foundation milestone

The branch now contains a staged HIP backend for the first AMD correctness
milestone. It does **not** claim full GPU inference.

- Build selection is explicit: `BACKEND=cpu` uses `gpu_offload_cpu.c`;
  `BACKEND=hip` compiles `gpu_offload_hip.cpp` with configurable `HIPCC` and
  `GPU_ARCH` (development default `gfx1030`). Ordinary runtime sources remain C.
- HIP initialization enumerates devices, deterministically selects device 0,
  reports name, `gcnArchName`, VRAM, runtime version and wave size, and creates
  separate non-blocking compute/H2D/D2H streams. Partial failures destroy any
  streams already created before returning failure.
- The first supported kernel execution model is wave32. Initialization compares
  the device-property and runtime-attribute wave sizes and rejects non-wave32
  devices diagnostically rather than applying CUDA's 32-lane assumptions.
- Q8_0 has three HIP implementations: single-thread reference, one-wave simple,
  and 256-thread/shared-input implementation. The 34-byte packed block's FP16
  scale is assembled bytewise and converted with HIP half intrinsics so the
  two-byte scale never requires an unaligned device load.
- `tests/hip_smoke_test.cpp` covers runtime/device query, allocation, async H2D,
  kernel launch, synchronization, async D2H, value checking and free.
- `tests/hip_q8_parity.cpp` builds deterministic independent CPU fixtures for
  zeros, positive/negative/alternating signs, int8 extrema, ordinary/tiny FP16
  scales, multiple rows and widths through 4096. It compares all three HIP
  kernels to a double-accumulating scalar CPU oracle and cross-compares the HIP
  implementations. The tolerance is fixed in source at
  `abs_error <= 5e-2 + 5e-5 * abs(reference)`.
- Runtime initialization is separated from full-inference capability. The HIP
  backend rejects full-inference capability and exposes staged GQA projections
  (Q/K/V/Wo) separately, so the unchecked DeltaNet async path remains on CPU. CUDA advertises full capability. This is
  required because several DeltaNet launch/wait/result calls do not have safe
  per-operation fallback semantics.
- The DeltaNet group-mapping discrepancy remains unresolved:
  `deltanet_impl.h` uses `h / DN_HEADS_PER_GROUP`, while the GPU-assisted host
  recurrence uses `h_idx % DN_NUM_KV_GROUPS`. No multi-group HIP DeltaNet work
  should use either path as an oracle until reference intermediates settle it.

Target-machine validation:

```sh
command -v hipcc
hipcc --version
rocminfo

make clean BACKEND=cpu
make BACKEND=cpu -j2
make BACKEND=cpu check

make clean BACKEND=hip
make BACKEND=hip GPU_ARCH=gfx1030 -j2
make BACKEND=hip GPU_ARCH=gfx1030 hip-check
```

If Arch exposes HIP outside `PATH`, override it explicitly, for example
`make BACKEND=hip HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1030 -j2`.

**Validation status:** verified on the target Arch Linux system with an
AMD Radeon RX 6950 XT / gfx1030. The HIP build completed, device 0 was identified
as gfx1030 with 16,368 MiB reported VRAM and wave32, the smoke test passed, and
all three Q8_0 HIP kernels passed the independent CPU parity suite. Full-model
HIP inference is still deliberately disabled.

## What builds and runs

- Native Linux C11/C++17/OpenMP CPU build with GCC and no Wine/CUDA/HIP
  dependency; optional HIP backend objects/tests are compiled and linked with
  ROCm `hipcc`.
- CPU executable `build/linux-cpu/linmoe`, HIP foundation executable
  `build/linux-hip/linmoe`, and metadata tool `linmoe-inspect`.
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

CPU sanitizer validation was originally performed in a Linux x86-64 container
with GCC 13 and AVX-512 host features. The CPU release regression suite and HIP
foundation were subsequently verified on the target Arch Linux machine with an
AMD Radeon RX 6950 XT / gfx1030. No throughput improvement or real-model
equivalence is claimed yet.

| Check | Result |
|---|---|
| `make BACKEND=cpu check` | Pass on target Arch system |
| GGUF synthetic cases | 133 pass, including every incomplete prefix of a valid fixture |
| Q6_K scalar versus AVX2 | 3,000 pairs pass; max abs 3.662109e-4, RMSE 4.061567e-5 |
| Q8 signed integer dot | All 65,536 constant int8 pairs plus mixed lanes pass exactly |
| POSIX reads | Unaligned payloads, exact EOF, sparse >4 GiB, post-open truncation, invalid ranges/fds pass |
| Analytic inference | GQA + DeltaNet, two positions, cold/hot cache, logits checked |
| HIP build | Pass with `BACKEND=hip GPU_ARCH=gfx1030` |
| HIP device diagnostics | RX 6950 XT; gfx1030; 16,368 MiB VRAM; wave32; HIP runtime 70253211 |
| HIP smoke | Pass: init, properties, allocation, H2D, kernel, D2H, validation, free |
| CPU vs HIP Q8 single | 7 cases / 42 rows; max abs 1.46484375e-3; RMSE 4.41727480e-4; max rel 5.19426715e-6 |
| CPU vs HIP Q8 simple | 7 cases / 42 rows; max abs 9.765625e-4; RMSE 2.52418728e-4; max rel 2.04786955e-6 |
| CPU vs HIP Q8 optimized | 7 cases / 42 rows; max abs 9.765625e-4; RMSE 2.33337389e-4; max rel 9.18743188e-7 |
| HIP single vs simple | Max abs 7.32421875e-4; RMSE 2.44237109e-4; max rel 3.63596812e-6 |
| HIP single vs optimized | Max abs 1.09863281e-3; RMSE 3.94883249e-4; max rel 4.88258576e-6 |
| GQA Q+gate | 36 calls; max abs 1.953125e-3; RMSE 4.16676675e-4 |
| GQA K | 36 calls; max abs 1.61743164e-3; RMSE 3.59322842e-4 |
| GQA V | 36 calls; max abs 1.46484375e-3; RMSE 3.46418818e-4 |
| GQA Wo | 30 calls; max abs 2.9296875e-3; RMSE 5.74647851e-4; max rel 6.55173437e-4 |
| GQA lifecycle | PASS: four-matrix replacement, Wo OOM/copy rollback, resize/reuse, shutdown |
| GQA hybrid inference | PASS: 3 positions; max logit abs 1.66893005e-6; RMSE 1.77150513e-7 |
| ASan + UBSan | Entire CPU `make check` suite passes with leak detection disabled |
| LeakSanitizer | Environment rejects its tracing/thread inspection; no leak-free claim |
| Real Qwen CPU/CUDA/HIP comparison | Not run |
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
| GPU boundary | `gpu_offload.h`; CPU declines initialization, CUDA advertises full inference, staged HIP initializes but advertises foundation-only capability |
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
| Portability backlog | Win32 APIs remain in legacy benchmarks/tests; HIP foundation exists, but no full HIP inference, io_uring or O_DIRECT implementation yet |

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
  lack complete error propagation. HIP foundation calls are checked and partial
  initialization cleans up correctly, but the inherited high-level CUDA-style
  inference API still needs equivalent per-operation error handling before full
  HIP inference can be enabled.
- Legacy Win32 read completion errors and synchronous-handle async fallback
  remain unsafe in the reference path; the native reader does not use them.
- CUDA has wave32 masks/reductions, fixed dimensions and launch geometries,
  fixed pipeline allocations, unchecked calls and incomplete shutdown coverage.
  HIP translation must review these, not just rename APIs.
- Expert cache entries and read pools assume uniform per-layer expert byte
  sizes. This baseline explicitly rejects mismatches; generalized sizing and
  request ownership belong in the subsequent streaming milestone.

## Next implementation order

1. Establish independent Q4_K and Q5_K CPU-vs-HIP block/matvec parity before
   touching expert kernels or cache residency. Reuse the Q8 test methodology:
   fixed deterministic fixtures, signed/extreme values, multiple blocks/rows,
   fixed tolerances and GPU cross-checks where useful.
2. Keep GQA attention on CPU for now; the validated Q/K/V/Wo projection boundary
   remains opt-in through `WINMOE_GQA_HIP=1`.
3. Resolve the DeltaNet mapping discrepancy and establish nonzero multi-group
   CPU/reference fixtures before using either existing recurrence path as the
   oracle for HIP DeltaNet work.
4. After Q4_K/Q5_K parity, port expert kernels/cache incrementally with explicit
   ownership, VRAM accounting and rollback tests.
5. Measure the buffered storage baseline, then add O_DIRECT alignment discovery
   and io_uring incrementally, with cancellation and buffer lifetime tests.
6. Query VRAM budgets with headroom, instrument residency and stalls, and measure
   disk/H2D/compute overlap before optimizing it.

Primary format/API references used for this baseline:
- https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- https://www.man7.org/linux/man-pages/man2/pread.2.html

Historical Windows/NVIDIA benchmark numbers elsewhere in this repository are
not LinMoE Linux/AMD measurements.
