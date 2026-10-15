"""Direct affine-domain lowering selects short physical ICU packets."""
import argparse
import csv
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
    for index, (counts, times, addresses) in enumerate([
        ((1, 1, 1), (1, 1, 1), (0, 0, 0)),
        ((4, 1, 1), (2, 1, 1), (-1, 0, 0)),
        ((1, 3, 2), (1, 2, 11), (0, 1, 8)),
        ((2, 3, 2), (1, 5, 17), (1, 8, 32)),
        ((4, 2, 3), (1, 4, 8), (1, 4, 8)),
    ]):
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
    expected = [('READ', 1, 1), ('READ_1D', 2, 4), ('READ_2D', 2, 6),
                ('READ_3D', 3, 12), ('READ_1D', 2, 24)]
    for i, (opcode, words, issues) in enumerate(expected):
        with (csv_dir / f'mem_E_slice{i:02d}_bank0.icu.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
        assert len(rows) == 1, rows  # wait belongs in the packet, no timing NOP
        row = rows[0]
        assert (row['opcode'], int(row['imem_words']), int(row['expanded_fu_issues'])) == (opcode, words, issues), row
        assert int(row['start_cycle']) == 100 and int(row['wait_cycle']) == 100, row
    print('direct short ICU lowering/physical CSV/wait encoding passed')


if __name__ == '__main__':
    main()
