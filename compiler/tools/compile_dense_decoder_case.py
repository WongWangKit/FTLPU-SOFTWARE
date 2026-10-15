#!/usr/bin/env python3
"""Compile the same full-size synthetic graph used by dense numerical tests."""
import argparse
from pathlib import Path
import subprocess
from create_dense_decoder_smoke_checkpoint import SPECS, model_config
from import_hf_decoder_layer import decoder_config
from run_official_dense_acceptance import source_ir


def source_ir_for_fixture(model):
    config, _, _ = model_config(model, SPECS[model])
    return source_ir(decoder_config(config), model)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=SPECS,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--target-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    source=a.output.with_suffix('.stablehlo.mlir')
    source.write_text(source_ir_for_fixture(a.model),encoding='utf-8')
    subprocess.run([str(a.compiler), '--input',str(source),'--output',str(a.output),
        '--target-config',str(a.target_config),'--weight-bank','1',
        '--ffn-schedule','fused','--attention-schedule','fused',
        '--projection-rope-overlap','off'],check=True)


if __name__=='__main__': main()
