"""Synthetic GGUF regression fixtures; no model download or tensor allocation.

The binary encoding below is independent of the production parser. Validate
exact sizes/offsets and rejection behavior, not generated text.
"""
import pathlib
import struct
import subprocess
import sys
import tempfile

INSPECT = str(pathlib.Path(sys.argv[1]).resolve())
COUNT = 0


def u32(n):
    return struct.pack('<I', n)


def u64(n):
    return struct.pack('<Q', n)


def string(s):
    b = s.encode()
    return u64(len(b)) + b


def metadata(key, kind, value):
    return string(key) + u32(kind) + value


def tensor(name='weight', dims=(32,), kind=8, offset=0):
    return string(name) + u32(len(dims)) + b''.join(map(u64, dims)) + u32(kind) + u64(offset)


def fixture(info=None, kv=(), alignment=32, payload=34, nt=1):
    """Default is one Q8_0 block ending exactly at EOF, without trailing padding."""
    if info is None:
        info = tensor()
    header = b'GGUF' + u32(3) + u64(nt) + u64(len(kv)) + b''.join(kv) + info
    return header + bytes((-len(header)) % alignment) + bytes(payload)


def run(path, valid, expected=()):
    global COUNT
    result = subprocess.run([INSPECT, str(path)], capture_output=True, text=True, timeout=5)
    assert result.returncode == (0 if valid else 1), (path, result.returncode, result.stdout, result.stderr)
    for item in expected:
        assert item in result.stdout, (item, result.stdout)
    if not valid:
        assert 'LinMoE:' in result.stderr, result.stderr
    COUNT += 1


with tempfile.TemporaryDirectory(prefix='linmoe-gguf-') as tmp:
    root = pathlib.Path(tmp)
    path = root / 'model.gguf'

    def check(data, valid=True, expected=()):
        path.write_bytes(data)
        run(path, valid, expected)

    check(fixture(), expected=('type=8 bytes=34', 'shards=1 tensors=1'))
    # Every incomplete prefix must fail, including headers and tensor payload.
    good = fixture()
    for n in range(len(good)):
        check(good[:n], False)
    values = [metadata('bool', 7, b'\x01'), metadata('u64', 10, u64(123)),
              metadata('i64', 11, u64(123)), metadata('f64', 12, struct.pack('<d', 2.5)),
              metadata('strings', 9, u32(8) + u64(2) + string('abc') + string('def')),
              metadata('qwen.rope.dimension_sections', 9, u32(4) + u64(17) + u32(1) * 17)]
    check(fixture(kv=values), expected=('bytes=34',))
    check(fixture(kv=[metadata('general.alignment', 4, u32(64))], alignment=64),
          expected=('offset=128',))
    check(fixture(kv=[metadata('general.alignment', 4, u32(3))]), False)
    check(fixture(kv=[metadata('general.alignment', 7, b'\x01')]), False)
    check(fixture(info=tensor(dims=())), False)
    check(fixture(info=tensor(dims=(1,) * 5)), False)
    check(fixture(info=tensor(dims=(0,))), False)
    check(fixture(info=tensor(dims=(1 << 63, 4))), False)
    check(fixture(info=tensor(dims=(31,))), False)
    check(fixture(info=tensor(kind=999)), False)
    check(fixture(info=tensor(offset=1), payload=35), False)
    check(fixture(info=tensor(offset=(1 << 64) - 32)), False)
    check(fixture(nt=4001), False)
    check(fixture(info=tensor() + tensor(), nt=2), False)
    check(fixture(kv=[metadata('bad-array', 9, u32(10) + u64(1 << 63))]), False)
    check(fixture(kv=[metadata('bad-kind', 55, b'')]), False)
    check(fixture(kv=[metadata('huge-string', 8, u64(1 << 63))]), False)
    check(fixture(info=tensor(name='x' * 256)), False)
    check(fixture(info=tensor(name='bad\0name')), False)
    check(b'BAD!' + good[4:], False)
    check(good[:4] + u32(99) + good[8:], False)
    # Confirm remaining supported block byte counts instead of an FP32 fallback.
    for kind, weights, size in [(0, 1, 4), (1, 1, 2), (2, 32, 18), (12, 256, 144),
                                (13, 256, 176), (14, 256, 210), (16, 256, 66)]:
        check(fixture(info=tensor(kind=kind, dims=(weights,)), payload=size),
              expected=(f'bytes={size}',))
    first = root / 'model-00001-of-00002.gguf'
    second = root / 'model-00002-of-00002.gguf'
    first.write_bytes(fixture())
    run(first, False)  # Missing shard must not succeed with a partial model.
    second.write_bytes(fixture(info=tensor(name='other')))
    run(first, True, ('shards=2 tensors=2', 'other type=8 bytes=34 shard=1'))
    run(second, False)  # Starting at shard 2 is a configuration error.
    second.write_bytes(fixture())
    run(first, False)  # Duplicate cross-shard names are ambiguous.
    too_many = root / 'model-00001-of-00009.gguf'
    too_many.write_bytes(fixture())
    run(too_many, False)
    # A sparse tensor beyond 4 GiB catches accidental 32-bit seek truncation.
    offset = 1 << 32
    data = fixture(info=tensor(offset=offset), payload=0)
    with path.open('wb') as f:
        f.write(data)
        f.truncate(len(data) + offset + 34)
    run(path, True, (f'offset={len(data) + offset}',))

print(f'GGUF regression tests PASS ({COUNT} cases)')
