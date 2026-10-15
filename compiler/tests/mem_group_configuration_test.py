"""Check that physical tile width and MEM/SR grouping lower independently."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--hardware-config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    config = json.loads(a.hardware_config.read_text(encoding='utf-8'))
    config['topology']['tiles_per_slice'] = 8
    config['target']['name'] = 'mem-group-independent-tiles8'
    config['stream_topology']['bindings']['mem']['slices_per_group'] = 4
    source = a.output_dir/'target.json'
    output = a.output_dir/'kernel.mlir'
    source.write_text(json.dumps(config),encoding='utf-8')
    command = [str(a.tool),'--input',str(a.input),'--output',str(output),
        '--pipeline','ftlpu-stablehlo-to-kernel','--target-config',str(source)]
    subprocess.run(command,check=True)
    lowered = output.read_text(encoding='utf-8')
    for field in ('tile_rows = 8 : i64','mem_slices_per_register_group = 4 : i64',
            'mem_boundary_register_columns = 14 : i64','system_register_columns = 16 : i64',
            'mxm_rows = 64 : i64','bytes_per_word = 64 : i64'):
        if field not in lowered: raise AssertionError('missing independent geometry: '+field)
    for group in (0,8):
        config['stream_topology']['bindings']['mem']['slices_per_group'] = group
        source.write_text(json.dumps(config),encoding='utf-8')
        result = subprocess.run(command,capture_output=True,text=True)
        if result.returncode==0 or 'MEM slices must divide' not in result.stderr:
            raise AssertionError(f'invalid MEM grouping was not rejected: {group}: {result.stderr}')
    print('mem_group_configuration_test passed: tiles=8 lanes=64 MEM groups=13')


if __name__=='__main__':main()
