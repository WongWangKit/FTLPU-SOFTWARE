from pathlib import Path
import hashlib, json, subprocess

sw=next(p for p in Path(__file__).resolve().parents if (p/'runtime/CMakeLists.txt').is_file())
root=sw/'out/icu-config'
cm=sw.parent/'FTLPU-CMODEL'
build=sw/'out/main-qctl-integration/build'
with (root/'binary-tests.log').open('w') as log:
    subprocess.run([build/'runtime/binary_queue_control_test.exe'],stdout=log,stderr=subprocess.STDOUT,check=True)
config=cm/'config/ftlpu-lpu32.json'
base=[str(build/'compiler/ftlpu-compile.exe'),'--input',str(sw/'compiler/examples/smollm2_135m_decoder_layer/decoder_layer_seq32.stablehlo.mlir'),
      '--input-stage','stablehlo','--weight-bank','1','--kv-cache-capacity','256','--mxm-execution','vector','--projection-rope-overlap','off','--ffn-schedule','tail']
result={'default_binary_comparisons':[], 'incompatible_overlays_rejected':[]}
for mode in ('none','raw-macro'):
    name='compiled-seq32-'+mode+'.ftlpu'
    args=base+['--target-config',str(config),'--icu-compression',mode,'--output',str(root/name)]
    with (root/('compile-'+mode+'.log')).open('w') as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    actual=(root/name).read_bytes()
    baseline=(sw/'out/qctl-16x8'/name).read_bytes()
    assert actual==baseline, 'default binary changed: '+mode
    result['default_binary_comparisons'].append({'mode':mode,'bytes':len(actual),'identical':True,'sha256':hashlib.sha256(actual).hexdigest()})
for key,value in [('mem_iq_depth',8),('c2c_imem_depth',16384),('fetch_latency_cycles',2),('macro_reservoir_words',4)]:
    profile=json.loads(config.read_text(encoding='utf-8'))
    profile['icu'][key]=value
    path=root/('mismatch-'+key+'.json')
    path.write_text(json.dumps(profile,indent=2),encoding='utf-8')
    proc=subprocess.run(base+['--target-config',str(path),'--output',str(root/'rejected.ftlpu')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output=proc.stdout.decode(errors='replace')
    (root/('mismatch-'+key+'.log')).write_text(output,encoding='utf-8')
    assert proc.returncode!=0 and 'ICU build parameter' in output and key in output, output
    result['incompatible_overlays_rejected'].append(key)
(root/'software-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
