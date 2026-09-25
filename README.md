# LinMoE

Active repository: **asdgdg355235/LinMoE**. Native Linux/AMD development targets
ROCm/HIP on RX 6950 XT / gfx1030. Upstream WinMoE is a reference only.

The first native **CPU foundation** builds with `make` and tests with `make check`.
It currently requires an AVX-512-capable x86-64 CPU. HIP, O_DIRECT and io_uring are
not implemented, and real-model Linux parity is not yet established.
See [Linux port status and audit](docs/linux/PORT_STATUS.md) for exact commands,
validation evidence, remaining correctness gates, and the implementation map.

---

The material below records inherited Windows/NVIDIA work; its results do not
represent measurements of this Linux/AMD port.

# WinMoE historical reference

**Running Qwen3.5-397B (397 billion parameters) on a consumer Windows laptop at 1.9 tok/s.**

This project adapts Apple's ["LLM in a Flash"](https://arxiv.org/abs/2312.11514) concepts and Dan Woods' [Flash-MoE](https://github.com/danveloper/flash-moe) (Mac/Metal) to Windows with discrete GPU, proving that SSD-streamed MoE inference is viable on consumer hardware.

## Results

| Configuration | tok/s | Hardware |
|--------------|-------|----------|
| Qwen3.5-397B, K=10, CPU cold | 0.44 | RTX 3070 Laptop 8GB, 40GB DDR4, Samsung 980 NVMe |
| Qwen3.5-397B, K=3, source-built AVX512 | **1.8** | Same |
| Qwen3.5-397B, K=3, source-built CUDA ngl=8 | **1.9** | Same |
| iPhone 17 Pro (reference) | 0.6 | Apple A19 Pro, 12GB unified |
| Flash-MoE Mac (reference) | 4.36 | M3 Max 48GB, 17.5 GB/s SSD |

**4.3x improvement** from 0.44 to 1.9 tok/s through systematic optimization — K reduction, GPU layer tuning, thread optimization, compiler upgrades — with no custom inference engine.

## Key Findings

1. **MoE expert streaming works on Windows** — a 397B model (107 GB at IQ2_XXS) runs interactively on a laptop with 40GB RAM and 8GB VRAM
2. **The bottleneck is I/O, not compute** — explicit unbuffered reads achieve 2,200 MB/s vs mmap's 586 MB/s (3.7x faster)
3. **K reduction is the most impactful lever** — reducing active experts from 10 to 3 with minimal quality loss at 2-bit quantization
4. **GPU offload is second-order** — only +6.6% improvement, confirming SSD I/O dominance
5. **Throughput improves with generation length** — page cache warming gives +24% free speedup
6. **Compute ceiling is 2.34 tok/s** — professor-validated formula: `TPS_max = 1 / (L × T_c)` at K=3

## How It Works

Mixture-of-Experts models activate only a fraction of their parameters per token. Qwen3.5-397B has 512 experts per layer but only uses 3-10 per token. This means most expert weights can live on SSD and stream into RAM/GPU on demand:

```
SSD (107 GB model) → RAM (expert cache) → GPU (active computation)
         ↑                    ↑                     ↑
    2.1 GB/s            19 GB/s              192 GB/s
    (bottleneck)        (fast)               (fast)
```

## Project Structure

```
├── STUDY.md                    # Full research narrative (ongoing)
├── CLAUDE.md                   # AI agent operating rules
├── PROFESSOR_QUESTIONS_V2.md   # Formal math questions with professor answers
│
├── docs/                       # Research specs, phases, gates
│   ├── checkpoints/            # Frozen baselines and milestones
│   ├── professor_answers_v2.md # Professor's formal answers
│   └── ...
│
├── engine/                     # Custom expert service engine
│   ├── benchmarks/             # I/O and integrated benchmarks
│   ├── io/                     # Explicit I/O path implementation
│   ├── replay/                 # Trace replay simulator
│   ├── cache/                  # Tiered cache manager
│   └── ...
│
├── work/                       # Experimental data
│   ├── baseline/               # Hardware benchmarks (SSD, PCIe)
│   ├── sim/                    # Cache simulator and results
│   └── traces/                 # Expert routing traces
│
├── loops/                      # Autoresearch experiment loops
│   ├── gpu-unblock/            # GPU layer sweep (22 trials)
│   ├── perf-397b/              # Quality frontier (K=3/4/5)
│   ├── engine-io/              # Custom I/O benchmarks
│   └── ...
│
└── experiments/                # Raw experiment data
    └── k_sweep/                # Full K sweep results
```

## Methodology

This project uses an **AI-driven autoresearch** approach inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch):

- Claude Code (Opus 4.6) runs as the autonomous research agent
- Each optimization is a single experiment: one change, one measurement, keep or reject
- All results tracked in append-only TSV ledgers
- Professor provides formal mathematical guidance on architecture decisions
- The V2 spec defines measurement tracks A-G with explicit gate criteria

## Key Experiments

### Hardware Characterization
- SSD I/O benchmark: explicit unbuffered (2.1 GB/s) vs mmap (586 MB/s) vs buffered (6.8 GB/s cached)
- PCIe transfer: pinned (19 GB/s) vs pageable (6-8 GB/s)
- Affine timing models: `t(x,m) = α + βm + x/B`

### Routing Trace Analysis
- OLMoE real inference: Zipf s=0.456, Gini=0.396, 37.5% consecutive overlap
- Gate-weight analysis: Zipf s=0.275 (synthetic inputs — conservative lower bound)
- Working set covers 96% of experts by 1000 tokens

### Cache Simulation
- Static LFU >> LRU (confirmed by professor)
- Tiered cache: VRAM (64% hit) + RAM (31% hit) + SSD (5% miss) = 95% combined
- Miss curve: `M(C) ≈ 1 - (C^0.54 - 1) / (512^0.54 - 1)`

### Quality Frontier
- K=3, K=4, K=5 all produce coherent output at IQ2_XXS quantization
- At 2-bit, quantization noise dominates expert-count noise (professor-confirmed)

### Custom Engine Benchmarks
- Explicit I/O: 3.7x faster than mmap on expert-sized blocks
- Integrated benchmark: projects 2.34 tok/s with pipeline overlap
- Page cache pre-staging: fails due to NVMe queue contention (professor-explained)

## Professor-Validated Theory

The professor provided rigorous answers with exact formulas. Key results:

- **Compute ceiling**: `TPS_max ≈ 1 / (L × T_c) = 2.34 tok/s` at K=3
- **I/O crossover**: `B_crossover = K × S_e / T_c = 1.39 GB/s` — explicit I/O (2.1 GB/s) is above crossover
- **NVMe contention**: supplementing mmap always degrades (M/G/1-PS model)
- **Cache cliff**: smooth miss curve, not phase transition — kink comes from `max(compute, I/O)`
- **Hardware ROI**: (1) replace mmap, (2) improve GPU compute, (3) faster SSD (last priority)
- **Prediction**: dead end — label entropy ~24.4 bits makes Fano bound prohibitive

## Requirements

- Windows 11
- NVIDIA GPU (8GB+ VRAM recommended)
- 32GB+ RAM (40GB+ for 397B)
- NVMe SSD with 200GB+ free
- Python 3.11+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (pre-built or source)
- Model: [unsloth/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF) (UD-IQ2_XXS recommended for 40GB RAM)

## Quick Start

```bash
# 1. Download model (~107 GB for IQ2_XXS)
huggingface-cli download unsloth/Qwen3.5-397B-A17B-GGUF \
  --include "UD-IQ2_XXS/*" --local-dir D:/models/qwen3.5-397b

# 2. Patch K (reduce active experts for speed)
python work/baseline/patch_k_experts.py <gguf_path> --k 3 --apply

# 3. Run inference
llama-cli --model <gguf_path> --n-gpu-layers 8 --ctx-size 512 \
  --threads 12 --n-predict 200 --prompt "What is quantum computing?"
```

## Related Work

- [LLM in a Flash](https://arxiv.org/abs/2312.11514) — Apple's foundational paper on SSD-streamed inference
- [Flash-MoE](https://github.com/danveloper/flash-moe) — Dan Woods' Mac/Metal implementation (4.36 tok/s on M3 Max)
- [autoresearch](https://github.com/karpathy/autoresearch) — Karpathy's AI experiment loop framework
- [KTransformers](https://github.com/kvcache-ai/ktransformers) — CPU-GPU heterogeneous MoE inference
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — The inference engine this project builds on

## Status

**Active research.** Currently working on:
- Source-built llama.cpp with CUDA + AVX512 optimization (1.9 tok/s achieved)
- Custom expert service engine to replace mmap with explicit I/O (benchmarks prove 3.7x speedup, integration in progress)
- Path to 2.34 tok/s compute ceiling via mmap replacement + GPU compute optimization
- V2 research report

## License

MIT

## Authors

- Idan T. ([@82Labs](https://82labs.io))
- Research conducted with Claude Code (Opus 4.6) as autonomous research agent
- Professor guidance on formal mathematical framework
