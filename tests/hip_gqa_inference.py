"""Nonzero synthetic inference: CPU attention with HIP Q/K/V/Wo projections.

Run after standalone Q8, GQA Q/K/V and Wo parity. Compare every emitted logit
and GQA trace summary; dedicated fixtures already check every projection row.
The optional --cpu-only mode validates the fixture/oracle locally without HIP.
"""
import math
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
sys.dont_write_bytecode = True
from inference_smoke import create_model

engine = str(pathlib.Path(sys.argv[1]).resolve())
cpu_only = '--cpu-only' in sys.argv[2:]
with tempfile.TemporaryDirectory(prefix='linmoe-gqa-inference-') as tmp:
    root = pathlib.Path(tmp)
    model = root / 'nonzero.gguf'
    create_model(model, gqa_nonzero=True)
    outputs = []
    traces = []
    # Repeat for each trace token so positions 0, 1 and 2 cover cache growth.
    for mode in ('reference', 'candidate'):
        for position in range(3):
            env = {k: v for k, v in os.environ.items() if not k.startswith('WINMOE_')}
            logits_path = root / f'{mode}-{position}.bin'
            trace_path = root / f'{mode}-{position}.tsv'
            env.update(OMP_NUM_THREADS='2', WINMOE_DUMP_ALL_LOGITS=str(logits_path),
                       WINMOE_TRACE_OUT=str(trace_path), WINMOE_TRACE_TOK=str(position))
            if mode == 'reference' or cpu_only:
                env.update(WINMOE_GQA_CPU='1', WINMOE_GQA_FP32_REFERENCE='1')
            else:
                env['WINMOE_GQA_HIP'] = '1'
            run = subprocess.run([engine, '--model', str(model), '--tokens', '3',
                                  '--prompt-tokens', '0,1'], cwd=root, env=env,
                                 capture_output=True, text=True, timeout=60)
            assert run.returncode == 0, (run.returncode, run.stderr)
            if mode == 'candidate' and not cpu_only:
                assert 'GPU GQA Q/K/V/Wo: 1 layers uploaded' in run.stderr, run.stderr
                assert 'GQA Q/K/V/Wo=enabled' in run.stderr, run.stderr
            assert f'Trace target: tok={position}' in run.stderr, run.stderr
            data = logits_path.read_bytes()
            assert len(data) == 3 * 32 * 4
            outputs.append(struct.unpack('<96f', data))
            selected = {}
            for line in trace_path.read_text().splitlines():
                fields = line.split('\t')
                if fields[0] == 'NODE' and fields[1] in (
                    'Qcur_full-0', 'Qcur_reshaped-0', 'gate_reshaped-0',
                    'attn_pregate-0', 'attn_gated-0', 'attn_output-0'):
                    selected[fields[1]] = [float(x) for x in fields[5:9]] + [float(x) for x in fields[9].split(',')]
            assert len(selected) == 6, (position, selected)
            assert selected['Qcur_full-0'][0] > 1e-3
            assert selected['attn_output-0'][0] > 1e-6
            traces.append(selected)
    # Tighter integration bound is fixed in advance for these small, bounded
    # weights. It is separate from the large dynamic-range standalone Q8 bound.
    errors = []
    for position in range(3):
        for actual, expected in zip(outputs[3+position], outputs[position]):
            assert math.isfinite(actual) and math.isfinite(expected)
            error = abs(actual-expected)
            assert error <= 1e-3 + 5e-5*abs(expected), (position, actual, expected)
            errors.append(error)
        for key in traces[position]:
            for actual, expected in zip(traces[3+position][key], traces[position][key]):
                assert math.isfinite(actual) and abs(actual-expected) <= 1e-3 + 5e-5*abs(expected), (position, key, actual, expected)
    label = 'CPU fixture self-check' if cpu_only else 'HIP Q/K/V/Wo + CPU attention parity'
    print(f'{label} PASS: 3 positions, nonzero GQA, max_abs_diff={max(errors):.9g}, '
          f'RMSE={math.sqrt(sum(x*x for x in errors)/len(errors)):.9g}')
