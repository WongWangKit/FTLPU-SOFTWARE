import subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
exe=root.parent/'main-common-barrier-ninja/runtime/binary_queue_control_test.exe'
with (root/'final-image-checks.log').open('w') as log:
 for seq in (32,128):
  for case in ('51200','25600','51200-jitter'):
   native=root/f'final-native-seq{seq}-{case}-linked.ftlpu'
   if not native.exists():native=root/f'final-native-seq{seq}-{case}.ftlpu.loaded.ftlpu'
   pseudo=root/f'final-pseudo-seq{seq}-{case}-linked.ftlpu'
   log.write(f'seq{seq} {case}\n');log.flush()
   subprocess.run([str(exe),'--compare-final',str(native),str(pseudo)],stdout=log,stderr=subprocess.STDOUT,check=True)
with (root/'final-load-checks.log').open('w') as log:
 for seq in (32,128):
  for mode in ('native','pseudo'):
   path=root/f'final-{mode}-seq{seq}-25600-linked.ftlpu'
   log.write(path.name+'\n');log.flush()
   subprocess.run([str(exe),'--validate-final',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
print('6 executable image comparisons and 4 final image loads passed')
