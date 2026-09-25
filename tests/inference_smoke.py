"""Run a small analytic GGUF through the actual CPU executable.

Zero attention/expert weights leave constant embeddings unchanged. A single
nonzero output row has a known RMSNorm-based logit. This exercises GQA,
DeltaNet, expert reads, cache hits, and shutdown, but is NOT real-model parity.
"""
import math
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

H, I, VOCAB, EXPERTS = 256, 256, 32, 3


def u32(n):
    return struct.pack('<I', n)


def u64(n):
    return struct.pack('<Q', n)


def string(s):
    data = s.encode()
    return u64(len(data)) + data


def create_model(path, gqa_nonzero=False):
    """Use actual packed block sizes, padding only between tensors."""
    metadata = {
        'embedding_length': H, 'expert_feed_forward_length': I,
        'block_count': 2, 'expert_count': EXPERTS, 'expert_used_count': 3,
        'attention.head_count': 1, 'attention.head_count_kv': 1,
        'ssm.state_size': 128, 'ssm.inner_size': 128, 'ssm.group_count': 1,
        'ssm.conv_kernel': 4,
    }
    entries, payload = [], bytearray()

    def add(name, dims, kind, data=None, one=False):
        block_weights, block_bytes = {0: (1, 4), 8: (32, 34), 12: (256, 144), 13: (256, 176)}[kind]
        count = math.prod(dims)
        if data is None:
            data = struct.pack('<f', 1.) * count if one else bytes(count // block_weights * block_bytes)
        assert len(data) == count // block_weights * block_bytes
        payload.extend(bytes((-len(payload)) % 32))
        entries.append(string(name) + u32(len(dims)) + b''.join(map(u64, dims)) + u32(kind) + u64(len(payload)))
        payload.extend(data)

    embedding = (struct.pack('<e', 1.) + bytes([1]) * 32) * (H * VOCAB // 32)
    if gqa_nonzero:
        # Distinct tokens and signed channels exercise multi-position attention.
        embedding = b''.join(struct.pack('<e', 0.125) +
            bytes(((j * 3 + b * 7) % 17 - 8) & 255 for j in range(32))
            for b in range(H * VOCAB // 32))
    add('token_embd.weight', (H, VOCAB), 8, embedding)
    add('output_norm.weight', (H,), 0, one=True)
    head = [0.] * (H * VOCAB)
    head[H:2 * H] = [1.] * H
    add('output.weight', (H, VOCAB), 0, struct.pack(f'<{len(head)}f', *head))
    for layer in range(2):
        prefix = f'blk.{layer}.'
        add(prefix + 'attn_norm.weight', (H,), 0, one=True)
        add(prefix + 'ffn_norm.weight', (H,), 0, one=True)
        add(prefix + 'ffn_gate_inp.weight', (H, EXPERTS), 0)
        add(prefix + 'ffn_gate_exps.weight', (H, I, EXPERTS), 12)
        add(prefix + 'ffn_up_exps.weight', (H, I, EXPERTS), 12)
        add(prefix + 'ffn_down_exps.weight', (I, H, EXPERTS), 13)
        if layer == 0:
            # Default analytic fixture is unchanged. The optional hybrid fixture
            # has two query heads, one KV head, and nonzero interleaved gate rows.
            hd = 128 if gqa_nonzero else 256
            def gqa_data(rows, cols, seed):
                if not gqa_nonzero:
                    return None
                return b''.join(struct.pack('<e', 1 / 512) +
                    bytes(((r * 13 + b * 3 + j * 7 + seed) % 19 - 9) & 255 for j in range(32))
                    for r in range(rows) for b in range(cols // 32))
            add(prefix + 'attn_q.weight', (H, 512), 8, gqa_data(512, H, 1))
            add(prefix + 'attn_k.weight', (H, hd), 8, gqa_data(hd, H, 3))
            add(prefix + 'attn_v.weight', (H, hd), 8, gqa_data(hd, H, 5))
            add(prefix + 'attn_output.weight', (256, H), 8, gqa_data(H, 256, 7))
            add(prefix + 'attn_q_norm.weight', (hd,), 0, one=True)
            add(prefix + 'attn_k_norm.weight', (hd,), 0, one=True)
        else:
            add(prefix + 'ssm_a', (1,), 0)
            add(prefix + 'ssm_dt.bias', (1,), 0)
            add(prefix + 'ssm_norm.weight', (128,), 0, one=True)
            add(prefix + 'ssm_conv1d.weight', (4, 384), 0)
            add(prefix + 'attn_qkv.weight', (H, 384), 8)
            add(prefix + 'attn_gate.weight', (H, 128), 8)
            add(prefix + 'ssm_out.weight', (128, H), 8)
            add(prefix + 'ssm_alpha.weight', (H, 1), 0)
            add(prefix + 'ssm_beta.weight', (H, 1), 0)
    header = b'GGUF' + u32(3) + u64(len(entries)) + u64(len(metadata))
    for key, value in metadata.items():
        header += string('qwen35moe.' + key) + u32(4) + u32(value)
    header += b''.join(entries)
    path.write_bytes(header + bytes((-len(header)) % 32) + payload)


def run_cpu_smoke(engine):
    ENGINE = str(pathlib.Path(engine).resolve())
    with tempfile.TemporaryDirectory(prefix='linmoe-inference-') as tmp:
        root = pathlib.Path(tmp)
        model = root / 'analytic.gguf'
        create_model(model)
        env = {key: value for key, value in os.environ.items() if not key.startswith('WINMOE_')}
        env.update(OMP_NUM_THREADS='2', WINMOE_DUMP_ALL_LOGITS=str(root / 'logits.bin'),
                   WINMOE_TRACE_OUT=str(root / 'trace.tsv'))
        run = subprocess.run([ENGINE, '--model', str(model), '--tokens', '2', '--prompt-tokens', '0'],
                             cwd=root, env=env, capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)
        assert 'Expert cache: 6 hits, 6 misses' in run.stderr, run.stderr
        data = (root / 'logits.bin').read_bytes()
        assert len(data) == 2 * VOCAB * 4
        logits = struct.unpack(f'<{2 * VOCAB}f', data)
        expected = H / math.sqrt(1 + 1e-6)
        for pos in range(2):
            for token in range(VOCAB):
                value = logits[pos * VOCAB + token]
                target = expected if token == 1 else 0.
                assert math.isfinite(value) and abs(value - target) < 1e-3, (pos, token, value, target)
        assert 'SENTINEL\t198' not in (root / 'trace.tsv').read_text()
        # CLI errors must return nonzero before silently selecting defaults.
        for args in [[], ['--tokens', '0'], ['--tokens', '513'], ['--tokens', 'bad'], ['--unknown']]:
            result = subprocess.run([ENGINE] + args, cwd=root, capture_output=True, text=True)
            assert result.returncode == 2, (args, result.returncode)

    print('Analytic CPU inference PASS (GQA + DeltaNet, 2 positions, cold/hot cache, logits)')


if __name__ == "__main__":
    run_cpu_smoke(sys.argv[1])
