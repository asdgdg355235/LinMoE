# HIP GQA Q/K/V/Wo implementation and validation handoff

Baseline: `91bbb6baf7c1111f636361f778d191149481fcea`, branch
`chatgpt/hip-foundation-gfx1030`, in the authoritative LinMoE repository.
The local session initially held only the CPU baseline. The existing HIP branch
was fetched and its incremental changes applied after verifying it preserved
all local CPU changes. The HIP foundation was recovered, not reimplemented.

Status: **Q/K/V is target-validated on the RX 6950 XT; Wo is implemented and
requires target validation.** The validated Q/K/V milestone completed 36
projection calls with max absolute differences 0.001953125 (Q+gate),
0.001617432 (K), and 0.001464844 (V). The three-position Q/K/V hybrid fixture
had zero logit difference. This environment still has no local `hipcc` or AMD
device, so the newly added Wo path is not claimed PASS until target output is
returned.

## Data flow and geometry

CPU RMSNorm → HIP Wq/Wk/Wv → CPU QK norm/RoPE/KV cache/attention/gating →
HIP Wo → existing residual path. Q/K/V and Wo reuse the same validated simple
wave32 Q8 kernel and compute stream; attention itself remains on CPU.

QK normalization, RoPE, KV cache append, score computation, softmax, value
accumulation, sigmoid gating, and output projection remain on CPU for HIP.
The full-inference capability remains false; DeltaNet and experts cannot enter
unimplemented GPU paths. A separate Q/K/V capability controls this boundary.

| Matrix | Mathematical shape (output × input) | Representative 397B geometry |
|---|---|---|
| Wq | `(2 * query_heads * head_dim) × hidden` | `16384 × 4096` |
| Wk | `(kv_heads * head_dim) × hidden` | `512 × 4096` |
| Wv | `(kv_heads * head_dim) × hidden` | `512 × 4096` |

The inherited upload API names are misleading: `*_rows` means GGUF `dims[0]`
(input width); `*_cols` means `dims[1]` (output count). That ABI is preserved.
The caller now uses loaded dimensions, replacing hard-coded 32-head/256-channel
upload geometry. It validates rank, Q8_0 type, input/output sizes, and packed
byte count before passing a host pointer to HIP. Representative sizes above
are fixtures, not measurements from an actual loaded 397B model in this session.

`q_gate` contains `[Q_head0(D), gate_head0(D), Q_head1(D), gate_head1(D), ...]`.
It is neither plain Q nor a whole-Q vector followed by a whole-gate vector.
The backend preserves every output row in order; the CPU deinterleaves it.

## Ownership and failures

Each layer index 0–63 owns Wq/Wk/Wv and, when supplied, Wo device allocations
until replacement or shutdown. Upload borrows host pointers only until its checked synchronization
finishes. Host model weights remain owned by the existing loader for CPU use.
Wo is now optional in the HIP upload API for compatibility with the established
Q/K/V-only fixture. When supplied, it is the fourth matrix in the same
transaction and maps `q_gate_dim/2` attention values to `hidden_dim` outputs.
`gpu_gqa_output()` uses the common Q8 launch helper and publishes host output
only after synchronization.

Checked packed bytes are `output * (input / 32) * 34`, using `size_t` and the
existing overflow helper. Input width must be positive and divisible by 32.
K/V output counts must agree and Q+gate must be an integer multiple of twice K.
The implementation does not assume a particular hidden width or head dimension.

Re-upload allocates/copies an entire replacement before freeing old weights.
OOM drains already-enqueued copies and frees only the replacement; old weights
remain usable. A test hook simulates failure at each of the three allocations.
A copy/launch/runtime failure drains the stream and disables further GQA work
until shutdown. Failure to drain or free is fatal rather than discarding memory
ownership or returning host buffers with outstanding work. Inference treats
GPU upload/projection failure as fatal and never consumes stale outputs.

One growable scratch allocation holds FP32 input plus Q/K/V outputs. A matching
host staging allocation is reused. The input is copied once per call, all work
uses the existing compute stream, and output publication occurs only after all
three results are ready. GQA calls are serialized on one host thread. Shutdown
frees every layer, scratch, staging, then streams. Device allocation accounting
includes persistent matrices and scratch; it excludes runtime/driver overhead.

At `H=4096, Q+gate=16384, K=V=512`, Q/K/V storage remains **72.25 MiB**
(75,759,616 bytes). Wo adds 35,651,584 bytes, for **106.25 MiB** (111,411,200
bytes) of four-matrix residency per layer. Q/K/V scratch remains the larger
requirement at 86,016 bytes; Wo alone needs 49,152 bytes, so the shared grow-only
scratch allocation does not increase for representative geometry.
These are calculated sizes, not measured VRAM. Replacement temporarily needs
both old and new weight storage. The GPU fixture verifies tracked usage.

HIP stream/host-memory semantics consulted:
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html

Host staging is ordinary pageable memory; copies may block internally. This
milestone relies on ordering and explicit synchronization, not async overlap.

## Validation and numerical contract

The three existing HIP Q8 kernels are unchanged. Their dispatch is shared between
the standalone tests and resident GQA; production GQA chooses the validated
simple wave32 kernel. The standalone fixture/oracle code was extracted into
`tests/hip_q8_fixture.h` without changing its cases or tolerances.

GQA tests independently report Q+gate, K and V max absolute error, RMSE, maximum
relative error (reference magnitude above 1e-6), and worst case/output row.
Their bound is inherited: `abs_error <= 0.05 + 0.00005 * abs(reference)`.
For uniformly subnormal weights, the absolute term is multiplied by the scale,
so a missing tiny result cannot pass just because its signal is below 0.05.
These bounds were fixed before any new target-device result was observed.

Shapes `(hidden, Q+gate, K=V)` are `(64,128,32)`, `(96,96,24)`,
`(256,256,64)`, and `(4096,16384,512)`. Coverage includes deterministic signed
weights, -128/+127, zero rows/blocks, ordinary and subnormal scales, changed
inputs, output guards, invalid calls, two independently resident layers,
replacement, allocation-failure rollback, scratch growth, two complete runtime
lifetimes, and idempotent shutdown. All gate rows are checked.

The nonzero inference fixture uses two query heads, one KV head, hd=128,
H=256, three positions and nonzero Wo. It compares all logits and GQA trace
summaries against a CPU FP32-input oracle. The default CPU fast path quantizes
activations to Q8_K for larger matvecs; its additional quantization error is not
HIP reduction error. `WINMOE_GQA_FP32_REFERENCE=1` selects the scalar oracle
only at the GQA boundary. Normal CPU inference is unchanged. Integration uses
the separately fixed bound `0.001 + 0.00005 * abs(reference)`.

| Check in this session | Result |
|---|---|
| Pre-change CPU suites, before and after recovering HIP baseline | PASS |
| Clean CPU build and full `make BACKEND=cpu check` | PASS |
| Q6 scalar/AVX2 | 3000 pairs; max abs 3.662109e-4; RMSE 4.061567e-5 |
| Q8 signed dot | 65536 constant pairs plus mixed lanes PASS |
| POSIX / GGUF / analytic inference | PASS / 133 cases PASS / two positions PASS |
| ASan+UBSan CPU suite and new nonzero fixture self-check | PASS, leak detection disabled |
| New fixture CPU self-check | PASS; max abs/RMSE 0 (CPU vs same CPU oracle, not GPU evidence) |
| Required-device negative check | PASS; CPU executable cannot masquerade as HIP |
| Host C++ syntax of Q8/GQA fixture sources | PASS; not a HIP backend compilation |
| New HIP build | BLOCKED: hipcc unavailable |
| Existing standalone Q8 regression on modified backend | NOT RUN here |
| New Q+gate parity metrics | PENDING target output |
| New K parity metrics | PENDING target output |
| New V parity metrics | PENDING target output |
| Hybrid inference / device lifetime tests | PENDING target execution |

The prior baseline's target results remain recorded in `PORT_STATUS.md`:
RX 6950 XT, gfx1030, wave32, 16,368 MiB, runtime 70253211; Q8 simple max abs
9.765625e-4, RMSE 2.52418728e-4. Those are historical results, not a rerun of
this change. No gfx1030 runtime issue was observed in this session because no
AMD device is available. Device errors require the target log to diagnose.

## Commands

From a checkout containing the baseline above, apply the incremental patch:

```sh
git apply --check /path/to/LinMoE-hip-gqa.patch
git apply /path/to/LinMoE-hip-gqa.patch
```

Preserve any newer local changes; do not force a failed patch application.
Then run on the RX 6950 XT:

```sh
command -v hipcc
hipcc --version
rocminfo
make clean
make BACKEND=cpu -j2
make BACKEND=cpu check
make BACKEND=hip GPU_ARCH=gfx1030 -j2
set -o pipefail
make BACKEND=hip GPU_ARCH=gfx1030 hip-check 2>&1 | tee hip-gqa-check.log
```

If needed, add `HIPCC=/actual/path/to/hipcc` to both HIP make commands. For
example `/opt/rocm/bin/hipcc --version` is appropriate only if that file exists.
No ROCm install location is hard-coded. Return the build and test output,
including device properties and all Q8 and Q/K/V metric lines.

`hip-check` runs smoke → standalone Q8 → GQA fixture → hybrid inference, stopping
at failure. Individual binaries are `build/linux-hip/hip-smoke-test`,
`build/linux-hip/hip-q8-parity`, and `build/linux-hip/hip-gqa-parity`; integration
is `python3 tests/hip_gqa_inference.py build/linux-hip/linmoe`.

For CPU-only inspection of the new nonzero fixture:

```sh
python3 tests/hip_gqa_inference.py build/linux-cpu/linmoe --cpu-only
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
make BACKEND=cpu BUILD=build/gqa-sanitize \
  CFLAGS='-O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  CXXFLAGS='-O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer' -j2 check
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
python3 tests/hip_gqa_inference.py build/gqa-sanitize/linmoe --cpu-only
```

After target parity passes, explicit hybrid inference is available with
`WINMOE_GQA_HIP=1 build/linux-hip/linmoe --model ...`. Presence of
`WINMOE_GQA_CPU` overrides GQA offload. Default HIP inference stays on CPU
until the new milestone is accepted. `WINMOE_TRACE_TOK=N` selects a checked
forward position for intermediate trace summaries without adding normal-run work.

## Exact files changed relative to the HIP baseline

- `Makefile`
- `docs/linux/PORT_STATUS.md`
- `docs/linux/HIP_GQA.md`
- `engine/runtime/gpu_offload.h`
- `engine/runtime/gpu_offload_cpu.c`
- `engine/runtime/gpu_offload.cu`
- `engine/runtime/gpu_offload_hip.cpp`
- `engine/runtime/gpu_offload_hip_test.h`
- `engine/runtime/winmoe_inference.c`
- `tests/hip_q8_fixture.h`
- `tests/hip_q8_parity.cpp`
- `tests/hip_gqa_parity.cpp`
- `tests/hip_gqa_inference.py`
- `tests/inference_smoke.py`

CPU/CUDA changes add the narrow capability; CUDA kernels are unchanged.
The Makefile also preserves CPU sanitizer flags during separate-object linkage.
The original zero-weight inference fixture's behavior remains unchanged.

## Remaining gates and next subsystem

The new HIP translation unit must compile and all GPU-required tests must pass
on gfx1030 before this milestone is complete. Full-model parity and performance
are unmeasured. Wave64, multi-threaded host access and multi-GPU scheduling remain
unsupported. No full GPU inference is advertised.

The DeltaNet mapping disagreement remains deferred: CPU uses
`h / DN_HEADS_PER_GROUP`, GPU-assisted host recurrence uses
`h_idx % DN_NUM_KV_GROUPS`. Resolve it with a pinned reference and a multi-group
regression before any HIP DeltaNet work. The current bounded subsystem is GQA Wo output projection using the same Q8
helper. After Wo target acceptance, complete-GQA correctness is established and
DeltaNet should remain deferred until its group-mapping discrepancy is resolved.
