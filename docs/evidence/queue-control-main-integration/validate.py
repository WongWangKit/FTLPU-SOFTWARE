import json, os, subprocess, sys
from pathlib import Path
sw=next(p for p in Path(__file__).resolve().parents if (p/'runtime/CMakeLists.txt').is_file())
root=sw/'out/main-qctl-integration';cm=sw.parent/'FTLPU-CMODEL'
build=root/'build'
def run(tag,args,env=None):
    with (root/(tag+'.log')).open('w') as f:
        code=subprocess.call(list(map(str,args)),env=env,stdout=f,stderr=subprocess.STDOUT)
    print(tag,code,flush=True)
    if code:
        print((root/(tag+'.log')).read_text(errors='replace')[-4000:],flush=True)
        raise SystemExit(code)
if sys.argv[1]=='tests':
    ctest=Path('D:/Visual Studio/Visual Studio 2022/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe')
    run('software-tests',[ctest,'--test-dir',build,'-R','^(runtime_common_barrier_test|binary_barrier_test|macro_bitstream_test|fu_3d_macro_lowering_test|model_session_executable_c2c_weight_test|issue_inspector_test|c2c_weight_ping_pong_test|binary_queue_control_test|program_load_runtime_test)$','--output-on-failure'])
elif sys.argv[1]=='compile':
    for seq in (32,128):
        for mode in ('none','raw-macro'):
            run(f'compile-seq{seq}-{mode}',[build/'compiler/ftlpu-compile.exe','--input',sw/f'compiler/examples/smollm2_135m_decoder_layer/decoder_layer_seq{seq}.stablehlo.mlir','--output',root/f'compiled-seq{seq}-{mode}.ftlpu','--input-stage','stablehlo','--target-config',cm/'config/ftlpu-lpu32.json','--weight-bank','1','--kv-cache-capacity','256','--mxm-execution','vector','--projection-rope-overlap','off','--ffn-schedule','tail','--icu-compression',mode])
elif sys.argv[1]=='run':
    seq=int(sys.argv[2]);bw=int(sys.argv[3])
    for mode in ('native','pseudo'):
        tag=f'{mode}-seq{seq}-{bw}'
        env=os.environ.copy()
        env.pop('FTLPU_QUEUE_CONTROL_FINAL_ENCODING',None)
        env.pop('FTLPU_DDR_EXTRA_READ_LATENCY',None)
        if mode=='native':env['FTLPU_QUEUE_CONTROL_FINAL_ENCODING']='native'
        env['FTLPU_DDR_BANDWIDTH_MBPS']=str(bw)
        env['FTLPU_QWEN_C2C_PIPELINE_CSV']=str(root/(tag+'.csv'))
        env['FTLPU_QWEN_C2C_LINKED_BINARY']=str(root/(tag+'-linked.ftlpu'))
        env['FTLPU_QUEUE_PROGRESS_CSV']=str(root/(tag+'-queues.csv'))
        args=[build/'runtime'/('compiled_smollm2_common_barrier_'+('seq128_' if seq==128 else '')+'decoder_test.exe'),root/f'compiled-seq{seq}-none.ftlpu']
        if mode=='pseudo':args.append(root/f'compiled-seq{seq}-raw-macro.ftlpu')
        run(tag,args,env)
