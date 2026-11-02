import os, subprocess, sys
from pathlib import Path

sw = next(p for p in Path(__file__).resolve().parents if (p / 'runtime/CMakeLists.txt').is_file())
root = sw / 'out/qctl-16x8'
root.mkdir(parents=True, exist_ok=True)
cm = sw.parent / 'FTLPU-CMODEL'
build = sw / 'out/main-qctl-integration/build'

def run(tag, args, env=None):
    with (root / (tag + '.log')).open('w') as log:
        result = subprocess.call(list(map(str, args)), env=env, stdout=log, stderr=subprocess.STDOUT)
    print(tag, result, flush=True)
    if result:
        print((root / (tag + '.log')).read_text(errors='replace')[-4000:], flush=True)
        raise SystemExit(result)

if sys.argv[1] == 'test':
    run('binary-tests', [build / 'runtime/binary_queue_control_test.exe'])
elif sys.argv[1] == 'compile':
    for mode in ('none', 'raw-macro'):
        run('compile-seq32-' + mode, [build / 'compiler/ftlpu-compile.exe',
            '--input', sw / 'compiler/examples/smollm2_135m_decoder_layer/decoder_layer_seq32.stablehlo.mlir',
            '--output', root / ('compiled-seq32-' + mode + '.ftlpu'),
            '--input-stage', 'stablehlo', '--target-config', cm / 'config/ftlpu-lpu32.json',
            '--weight-bank', '1', '--kv-cache-capacity', '256', '--mxm-execution', 'vector',
            '--projection-rope-overlap', 'off', '--ffn-schedule', 'tail', '--icu-compression', mode])
elif sys.argv[1] == 'run':
    for mode in ('native', 'pseudo'):
        tag = mode + '-seq32-25600'
        # Avoid inherited trace-only/numeric-skip or alternate scheduling settings.
        env = {k: v for k, v in os.environ.items() if not k.startswith('FTLPU_')}
        if mode == 'native':
            env['FTLPU_QUEUE_CONTROL_FINAL_ENCODING'] = 'native'
        env['FTLPU_DDR_BANDWIDTH_MBPS'] = '25600'
        env['FTLPU_QWEN_C2C_PIPELINE_CSV'] = str(root / (tag + '.csv'))
        env['FTLPU_QWEN_C2C_LINKED_BINARY'] = str(root / (tag + '-linked.ftlpu'))
        env['FTLPU_QUEUE_PROGRESS_CSV'] = str(root / (tag + '-queues.csv'))
        args = [build / 'runtime/compiled_smollm2_common_barrier_decoder_test.exe',
                root / 'compiled-seq32-none.ftlpu']
        if mode == 'pseudo':
            args.append(root / 'compiled-seq32-raw-macro.ftlpu')
        run(tag, args, env)
else:
    raise SystemExit('expected test, compile, or run')
