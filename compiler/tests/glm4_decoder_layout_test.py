"""Full-size GLM-4 seq32 dense graph and initialized-mask SRAM regression."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_official_dense_acceptance import source_ir
from import_hf_decoder_layer import decoder_config
from qwen2_5_1_5b_decoder_layer_ir_test import find_dictionary, parse_placement


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tool', type=Path, required=True)
    p.add_argument('--target-config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config = decoder_config(dict(model_type='chatglm', hidden_size=4096,
        ffn_hidden_size=13696, num_attention_heads=32, multi_query_group_num=2,
        kv_channels=128, num_layers=40, rmsnorm=True, multi_query_attention=True,
        add_qkv_bias=True, add_bias_linear=False, layernorm_epsilon=1.5625e-7))
    source = a.output_dir/'decoder.stablehlo.mlir'
    output = a.output_dir/'decoder.tensor.mlir'
    source.write_text(source_ir(config, 'glm4_9b_layout'), encoding='utf-8')
    subprocess.run([str(a.tool), '--input', str(source), '--output', str(output),
        '--target-config', str(a.target_config), '--weight-bank', '1',
        '--projection-rope-overlap', 'off', '--mxm-execution', 'ws',
        '--pipeline', 'ftlpu-stablehlo-to-tensor'], check=True)
    ir = output.read_text(encoding='utf-8')
    bank_rows = json.loads(a.target_config.read_text(encoding='utf-8-sig'))['mem']['rows_per_bank']
    for name in ('causal_mask', 'causal_mask_mxm1'):
        mask = parse_placement(find_dictionary(ir, name))
        assert 1024 <= mask['base'] and mask['base'] + mask['count'] <= bank_rows, mask
        for other in ('query_bias', 'key_bias', 'value_bias'):
            bias = parse_placement(find_dictionary(ir, other))
            if mask['bank'] == bias['bank'] and set(mask['slices']) & set(bias['slices']):
                assert mask['base'] + mask['count'] <= bias['base'] or bias['base'] + bias['count'] <= mask['base']
    assert 'rotary_dim = 64' in ir
    assert '13696' in ir and 'ftlpu.tensor.projection_task' in ir
    norms = [line for line in ir.splitlines() if 'ftlpu.tensor.rms_norm_task' in line]
    assert len(norms) == 2 and all('materialize_gamma = true' in line for line in norms)
    print('GLM-4 seq32 dense tensor layout passed')


if __name__ == '__main__':
    main()
