"""Seal completed legacy runner outputs from their success log for safe resume."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--runner',type=Path,required=True)
    a=p.parse_args()
    runner=a.runner.resolve();stat=runner.stat();count=0
    for line in a.log.read_text(encoding='utf-8').splitlines():
        match=re.fullmatch(r'(\d+) (\S+): (cycles=\d+ outputs=(\d+))',line)
        if not match: continue
        index,name,execution,outputs=match.groups()
        prefix=a.directory/f'{int(index)-1:03d}_{name}'
        binary=prefix.with_suffix('.ftlpu')
        paths=Path(f'{prefix}.inputs.txt').read_text(encoding='utf-8').splitlines()
        digest=hashlib.sha256(binary.read_bytes())
        digest.update((str(runner)+str(stat.st_mtime_ns)+str(stat.st_size)).encode())
        for path in paths: digest.update(hashlib.sha256(Path(path).read_bytes()).digest())
        hashes=[hashlib.sha256(Path(f'{prefix}.{i}.bin').read_bytes()).hexdigest() for i in range(int(outputs))]
        Path(f'{prefix}.execution.json').write_text(json.dumps(dict(identity=digest.hexdigest(),
            execution=execution+'\n',outputs=hashes,recorded_from_log=str(a.log)),indent=2)+'\n',encoding='utf-8')
        count+=1
    print(f'sealed {count} completed CModel executions')


if __name__=='__main__': main()
