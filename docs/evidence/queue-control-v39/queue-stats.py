import csv,collections
from pathlib import Path
root=Path(__file__).resolve().parent
names=['MEM','MXM Load','MXM Compute','MXM Dequant','VXM','SXM Transpose','SXM Permute','DMA','TX','RX']
for seq in (32,128):
 for mode in ('native','pseudo'):
  groups=collections.defaultdict(list)
  for r in csv.DictReader((root/f'final-{mode}-seq{seq}-25600-queues.csv').open()):
   groups[int(r['kind'])].append(int(r['wait_cycles']))
  print(seq,mode,[(names[k],len(v),min(v),max(v)) for k,v in groups.items()])
