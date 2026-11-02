import subprocess,hashlib,sys
from pathlib import Path
root=Path(__file__).resolve().parent
sw=Path(r'C:/Users/Jiang_Hua/.codex/worktrees/port-seq128-correctness/FTLPU-SOFTWARE')
exe=root.parent/'main-common-barrier-ninja/compiler/ftlpu-compile.exe'
target=Path(r'E:/Fudan_University/internship/timesintelli/LPU/work/FTLPU-CMODEL/out/main-mature-macro/config/ftlpu-lpu32.json')
for seq in (32,128):
 output=root/f'recompiled-seq{seq}-raw-macro.ftlpu'
 args=[str(exe),'--input',str(sw/f'compiler/examples/smollm2_135m_decoder_layer/decoder_layer_seq{seq}.stablehlo.mlir'),'--output',str(output),'--input-stage','stablehlo','--target-config',str(target),'--weight-bank','1','--kv-cache-capacity','256','--mxm-execution','vector','--projection-rope-overlap','off','--ffn-schedule','tail','--icu-compression','raw-macro']
 with (root/f'recompiled-seq{seq}-raw-macro.log').open('w') as log:
  status=subprocess.call(args,stdout=log,stderr=subprocess.STDOUT)
 if status:sys.exit(status)
 original=root/f'compiled-seq{seq}-raw-macro.ftlpu'
 assert original.read_bytes()==output.read_bytes(),'compiler output changed'
 print(seq,'compiler candidate byte-identical',hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)
