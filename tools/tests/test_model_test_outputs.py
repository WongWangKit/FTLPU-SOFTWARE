"""Semantic output-selection and diagnostic isolation regression tests."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'compiler/tools'))
from model_test_outputs import OutputContract, TestOutput, execute_with_contract


class OutputContractTests(unittest.TestCase):
    def test_only_diagnostics_can_be_omitted(self):
        contract = OutputContract(*(TestOutput(name, role) for name,role in (
            ('context','diagnostic'), ('key','persistent_state'),
            ('hidden','result'), ('next_input','handoff'))))
        self.assertEqual(contract.indices(), [1,2,3])
        self.assertEqual(contract.indices(True), [0,1,2,3])
        self.assertEqual(contract.attribute(), 'ftlpu.external_output_indices = [1 : i64, 2 : i64, 3 : i64]')
        with self.assertRaises(ValueError): contract.describe([None])
        with self.assertRaises(ValueError): TestOutput('key','unknown')
        with self.assertRaises(ValueError): OutputContract(TestOutput('x','result'),TestOutput('x','diagnostic'))

    def test_diagnostic_time_never_replaces_primary_and_stale_files_are_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'actual.0.bin').write_bytes(b'stale')
            contract = OutputContract(TestOutput('context','diagnostic'), TestOutput('kv','persistent_state'))
            tensors = [SimpleNamespace(shape=(32,32),dtype='bf16')]*2
            calls = []
            def once(directory, attrs, indices):
                calls.append((directory.name,indices))
                (directory/'graph_plan.json').write_text(json.dumps(dict(
                    external_output_indices=indices, nodes=[dict(nominal_end=10)])))
                (directory/'decoder_layer.schedule.mlir').write_text('ftlpu.schedule.vxm cycle = 1 : i64\n')
                for i in indices: (directory/f'actual.{i}.bin').write_bytes(bytes([i]))
                return [f'device{i}' for i in indices], dict(output_indices=indices,
                    timing=dict(runtime_cycles=100 if directory.name=='diagnostic' else 40))
            values, phase = execute_with_contract(contract,tensors,folder,once)
            self.assertEqual(values,['device0','device1'])
            self.assertEqual(phase['timing']['runtime_cycles'],40)
            self.assertEqual(phase['diagnostic_phase']['timing']['runtime_cycles'],100)
            self.assertFalse((folder/'actual.0.bin').exists())
            self.assertTrue((folder/'actual.1.bin').exists())
            self.assertEqual(calls, [('diagnostic',[0,1]),(folder.name,[1])])

    def test_ignored_compiler_policy_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            def once(directory, attrs, indices):
                (directory/'graph_plan.json').write_text('{"external_output_indices": []}')
                return [], dict(output_indices=[])
            with self.assertRaisesRegex(AssertionError,'compiler did not apply'):
                execute_with_contract(OutputContract(TestOutput('kv','persistent_state')),
                    [SimpleNamespace(shape=(1,32),dtype='bf16')],folder,once)
            self.assertEqual(json.loads((folder/'output_contract.json').read_text())['status'],'failed')


if __name__ == '__main__': unittest.main()
