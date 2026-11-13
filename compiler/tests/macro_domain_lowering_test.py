"""Direct affine-domain lowering emits packed Macro images and preserves exact issues."""
import argparse
import csv
import re
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compile', required=True)
    p.add_argument('--export', required=True)
    p.add_argument('--target-config', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    operations = []
    # Time/address domains are deliberately not foldable except the last one.
    domains = [
        ((1, 1, 1), (1, 1, 1), (0, 0, 0)),
        ((4, 1, 1), (2, 1, 1), (-1, 0, 0)),
        ((1, 3, 2), (1, 2, 11), (0, 1, 8)),
        ((2, 3, 2), (1, 5, 17), (1, 8, 32)),
        ((4, 2, 3), (1, 4, 8), (1, 4, 8)),
    ]
    for index, (counts, times, addresses) in enumerate(domains):
        operations.append(f'''ftlpu.schedule.mem_transfer {{
          address = 128 : i64, address_stride = {addresses[0]} : i64,
          cycle = 100 : i64, hemisphere = 0 : i64, slice = {index} : i64,
          opcode = "read", packed_stream = {index * 2} : i64,
          repeat_count = {counts[0]} : i64, repeat_interval = {times[0]} : i64,
          wave_count = {counts[1]} : i64, wave_interval = {times[1]} : i64,
          wave_address_stride = {addresses[1]} : i64,
          group_count = {counts[2]} : i64, group_interval = {times[2]} : i64,
          group_address_stride = {addresses[2]} : i64
        }}''')
    source = args.output_dir / 'short.schedule.mlir'
    source.write_text('module { func.func @short() {\n' + '\n'.join(operations) + '\nreturn\n}}', encoding='utf-8')
    binary = args.output_dir / 'short.ftlpu'
    subprocess.run([args.compile, '--input', str(source), '--input-stage', 'schedule',
                    '--output', str(binary), '--target-config', args.target_config], check=True)
    csv_dir = args.output_dir / 'icu_programs'
    subprocess.run([args.export, str(binary), str(csv_dir)], check=True)
    for i, (counts, times, addresses) in enumerate(domains):
        path=csv_dir / f'mem_E_slice{i:02d}_bank0.icu.csv'
        text=path.read_text(encoding='utf-8-sig')
        rows=list(csv.DictReader(line for line in text.splitlines() if not line.startswith('#')))
        expected=sorted((100+x*times[0]+y*times[1]+z*times[2],
                         128+x*addresses[0]+y*addresses[1]+z*addresses[2])
                        for z in range(counts[2]) for y in range(counts[1]) for x in range(counts[0]))
        actual=[]
        for row in rows:
            assert row['opcode']=='MEM_READ_MACRO',row
            assert row['raw_words'].startswith('file:'),row
            image=csv_dir/row['raw_words'][5:]
            assert image.is_file() and image.stat().st_size % 12 == 0,image
            details=dict(re.findall(r'(\w+)=([^ ]+)',row['details']))
            for y in range(int(row['count1'])):
                for x in range(int(row['count0'])):
                    actual.append((int(row['start_cycle'])+x*int(row['cycle_stride0'])+y*int(row['cycle_stride1']),
                                   int(details['address_base'])+x*int(details['induction_stride0'])+y*int(details['induction_stride1'])))
        assert sorted(actual)==expected,(i,actual,expected)
        assert len(rows)==([1,1,1,2,1][i]),(i,len(rows))
        assert int(rows[0]['wait_cycle'])==100,rows[0]
    print('direct Macro domains, physical images, exact cycles and addresses passed')


if __name__ == '__main__':
    main()
