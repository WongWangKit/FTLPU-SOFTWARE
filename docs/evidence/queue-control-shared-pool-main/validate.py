import os, subprocess, sys
from pathlib import Path

sw = next(p for p in Path(__file__).resolve().parents if (p/'runtime/CMakeLists.txt').is_file())
root = sw/'out/shared-pool-main'; root.mkdir(parents=True,exist_ok=True)
cm = sw.parent/'FTLPU-CMODEL'
build = sw/'out/main-qctl-integration/build'
seq = int(sys.argv[2]) if len(sys.argv)>2 else 32

def run(tag,args,env=None):
    with (root/(tag+'.log')).open('w') as f:
        code=subprocess.call(list(map(str,args)),env=env,stdout=f,stderr=subprocess.STDOUT)
    print(tag,code,flush=True)
    if code:
        print((root/(tag+'.log')).read_text(errors='replace')[-4000:],flush=True)
        raise SystemExit(code)

if sys.argv[1]=='test':
    run('binary-tests',[build/'runtime/binary_queue_control_test.exe'])
elif sys.argv[1]=='compile':
    for mode in ('none','raw-macro'):
        run(f'compile-seq{seq}-{mode}',[build/'compiler/ftlpu-compile.exe','--input',
            sw/f'compiler/examples/smollm2_135m_decoder_layer/decoder_layer_seq{seq}.stablehlo.mlir',
            '--output',root/f'compiled-seq{seq}-{mode}.ftlpu','--input-stage','stablehlo',
            '--target-config',cm/'config/ftlpu-lpu32.json','--weight-bank','1','--kv-cache-capacity','256',
            '--mxm-execution','vector','--projection-rope-overlap','off','--ffn-schedule','tail',
            '--icu-compression',mode])
elif sys.argv[1]=='run':
    for mode in ('native','pseudo'):
        tag=f'{mode}-seq{seq}-25600'
        env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
        if mode=='native':env['FTLPU_QUEUE_CONTROL_FINAL_ENCODING']='native'
        env['FTLPU_DDR_BANDWIDTH_MBPS']='25600'
        env['FTLPU_QWEN_C2C_PIPELINE_CSV']=str(root/(tag+'.csv'))
        env['FTLPU_QWEN_C2C_LINKED_BINARY']=str(root/(tag+'-linked.ftlpu'))
        env['FTLPU_QUEUE_PROGRESS_CSV']=str(root/(tag+'-queues.csv'))
        exe='compiled_smollm2_common_barrier_'+('seq128_' if seq==128 else '')+'decoder_test.exe'
        args=[build/'runtime'/exe,root/f'compiled-seq{seq}-none.ftlpu']
        if mode=='pseudo':args.append(root/f'compiled-seq{seq}-raw-macro.ftlpu')
        run(tag,args,env)
else:
    raise SystemExit('expected test, compile, or run')
