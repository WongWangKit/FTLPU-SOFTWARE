"""Output contracts for model tests; layout and transfer generation stay in C++.

An invocation boundary matters: a tensor consumed by another host-launched
invocation is a handoff, even when it is intermediate in the mathematical model.
Only diagnostic roots may be omitted from the primary execution.
"""
from dataclasses import dataclass
import hashlib
import json
import re


@dataclass(frozen=True)
class TestOutput:
    name: str
    purpose: str  # result, persistent_state, handoff, diagnostic

    def __post_init__(self):
        if not self.name or self.purpose not in {
                'result', 'persistent_state', 'handoff', 'diagnostic'}:
            raise ValueError(f'invalid test output: {self}')


class OutputContract:
    def __init__(self, *outputs):
        self.outputs = tuple(outputs)
        if len({o.name for o in outputs}) != len(outputs):
            raise ValueError('duplicate output name')

    def indices(self, diagnostic=False):
        return [i for i, o in enumerate(self.outputs)
                if diagnostic or o.purpose != 'diagnostic']

    def attribute(self, diagnostic=False):
        return 'ftlpu.external_output_indices = [' + ', '.join(
            f'{i} : i64' for i in self.indices(diagnostic)) + ']'

    def describe(self, tensors=None):
        if tensors is not None and len(tensors) != len(self.outputs):
            raise ValueError('output contract does not cover every graph result')
        result = []
        for i, output in enumerate(self.outputs):
            item = dict(index=i, name=output.name, purpose=output.purpose,
                        primary_ddr=output.purpose != 'diagnostic')
            if tensors is not None:
                item.update(shape=list(tensors[i].shape), dtype=tensors[i].dtype)
            result.append(item)
        return dict(version=1, policy='required', outputs=result,
                    external_output_indices=self.indices(),
                    diagnostic_output_indices=self.indices(True))


def verify_compute_schedule(folder, diagnostic):
    """Prove that diagnostic export did not change the arithmetic invocation."""
    plan = json.loads((folder/'graph_plan.json').read_text())
    other = json.loads((diagnostic/'graph_plan.json').read_text())
    if plan['nodes'] != other['nodes']:
        raise AssertionError('output policy changed native graph scheduling')
    cutoff = max(n.get('nominal_end', 0) for n in plan['nodes'])

    def instructions(path):
        result = []
        for line in (path/'decoder_layer.schedule.mlir').read_text().splitlines():
            if 'ftlpu.schedule.' not in line or 'ftlpu.schedule.binding' in line:
                continue
            cycles = re.findall(r'(?:cycle|start) = (-?\d+) : i64', line)
            if cycles and min(map(int, cycles)) <= cutoff:
                result.append(line)
        return result

    normal, reference = instructions(folder), instructions(diagnostic)
    if not normal or normal != reference:
        raise AssertionError('output policy changed native compute instructions')
    return dict(operations=len(normal), last_native_end=cutoff,
                sha256=hashlib.sha256('\n'.join(normal).encode()).hexdigest())


def execute_with_contract(contract, tensors, folder, run_once):
    """Return validation tensors plus PRIMARY timing, never diagnostic timing.

    run_once(directory, module_attrs, indices) must compile and execute a fresh
    raw-macro program and return (selected tensors, phase record). Required
    outputs are bit-compared across both runs. No CPU result replaces a device
    output. A failed/reused run cannot inherit a previous success marker.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'output_contract.json'
    record = dict(status='running', **contract.describe(tensors))
    path.write_text(json.dumps(record, indent=2)+'\n')

    def invoke(directory, diagnostic):
        indices = contract.indices(diagnostic)
        directory.mkdir(parents=True, exist_ok=True)
        for i in contract.indices(True):
            (directory/f'actual.{i}.bin').unlink(missing_ok=True)
        values, phase = run_once(directory, (contract.attribute(diagnostic),), indices)
        plan = json.loads((directory/'graph_plan.json').read_text())
        if plan.get('external_output_indices') != indices:
            raise AssertionError('compiler did not apply the requested output contract')
        if phase['output_indices'] != indices or len(values) != len(indices):
            raise AssertionError('runtime outputs disagree with the output contract')
        for i in contract.indices(True):
            if (directory/f'actual.{i}.bin').exists() != (i in indices):
                raise AssertionError(f'unexpected or missing DDR output {i}')
        return values, phase

    try:
        selected = contract.indices()
        diagnostic_phase = None
        if selected != contract.indices(True):
            values, diagnostic_phase = invoke(folder/'diagnostic', True)
        primary, phase = invoke(folder, False)
        timing = phase.get('timing', {})
        if 'output_readback_including_diagnostics_cycles' in timing:
            timing['required_output_readback_cycles'] = timing['output_readback_including_diagnostics_cycles']
            timing['isolated_diagnostic_readback_cycles'] = 0
        if diagnostic_phase is not None:
            for field in ('semantic_graph_sha256', 'compiler_sha256'):
                if field in phase and phase[field] != diagnostic_phase.get(field):
                    raise AssertionError(f'diagnostic execution has different {field}')
            proof = verify_compute_schedule(folder, folder/'diagnostic')
            for index in selected:
                if (folder/f'actual.{index}.bin').read_bytes() != (
                        folder/'diagnostic'/f'actual.{index}.bin').read_bytes():
                    raise AssertionError(f'diagnostic execution changed required output {index}')
            for index, value in zip(selected, primary):
                values[index] = value
            phase.update(diagnostic_phase=diagnostic_phase,
                         diagnostic_directory=str(folder/'diagnostic'),
                         arithmetic_schedule_identical_to_diagnostic=True,
                         compute_schedule_proof=proof)
        else:
            values = primary
        phase['output_contract'] = contract.describe(tensors)
        record.update(status='executed', phase=phase,
                      numerical_validation='caller must validate returned device tensors')
        return values, phase
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        path.write_text(json.dumps(record, indent=2)+'\n')
