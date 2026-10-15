"""Offline regression tests for orchestration, failure handling and coverage labels."""
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'run_model_tests.py'
spec = importlib.util.spec_from_file_location('model_tests', SCRIPT)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = self.root / 'hardware.json'
        self.target.write_text('{"clock":500}', encoding='utf-8')
        self.args = SimpleNamespace(all=False, model=None, suite='prefill',
            build_dir=self.root/'build with spaces', config='Release',
            target_config=self.target, output_root=self.root/'results',
            checkpoint_root=self.root/'checkpoints', checkpoint=None,
            download=False, allow_gated_skip=False, no_pipeline=True, dry_run=False)
        self.root_patch = patch.object(entry, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_unsupported_models_cannot_be_run_or_selected_in_batch(self):
        self.args.model = ['deepseek_native']
        with self.assertRaisesRegex(ValueError, 'does not support'):
            entry.selected_models(self.args)
        self.args.all = True
        self.assertNotIn('deepseek_native', entry.selected_models(self.args))
        self.assertNotIn('glm5_native', entry.selected_models(self.args))

    def test_exact_checkpoint_cannot_be_applied_to_multiple_models(self):
        self.args.all = True
        self.args.checkpoint = self.root/'one checkpoint'
        with self.assertRaisesRegex(ValueError, 'one model'):
            entry.selected_models(self.args)

    def test_exact_official_checkpoint_and_output_do_not_depend_on_directory_name(self):
        self.args.suite = 'official'
        self.args.checkpoint = self.root/'my locally verified weights'
        self.args.no_pipeline = True
        out = entry.result_directory(self.args, 'glm_edge_1_5b')
        plan = entry.commands(self.args, 'glm_edge_1_5b', out, self.root/'work')
        command = plan[0][1]
        self.assertEqual(command[command.index('--model-dir')+1], str(self.args.checkpoint))
        self.assertEqual(command[command.index('--result-dir')+1], str(out))
        self.assertIn('--no-pipeline', command)

    def test_prefill_compiles_directly_without_layout_regression_assertions(self):
        plan = entry.commands(self.args, 'qwen2_5_1_5b', self.root/'out', self.root/'work')
        self.assertEqual([phase for phase, _ in plan], ['compile', 'cmodel'])
        self.assertEqual(plan[0][1][0], str(entry.executable(self.args.build_dir,'Release','compile')))
        self.assertEqual(plan[0][1][-1], 'on')
        # The numerical harness validates K/V state as well as the layer output.
        command = plan[0][1]
        self.assertEqual(command[command.index('--kv-cache-capacity') + 1], '256')
        self.assertFalse(any('pipeline_test.py' in arg for _, cmd in plan for arg in cmd))

    def test_local_pinned_checkpoint_is_not_redownloaded(self):
        self.args.suite = 'official'
        self.args.download = True
        checkpoint = entry.checkpoint_directory(self.args,'phi4_mini')
        checkpoint.mkdir(parents=True)
        for name in ('config.json','model.safetensors','provenance.json'):
            (checkpoint/name).write_text('{}')
        self.assertEqual(entry.commands(self.args,'phi4_mini',self.root/'out',self.root/'work')[0][0], 'official')

    def test_bypass_and_hardware_override_flags_are_removed(self):
        with patch.dict(os.environ, {'FTLPU_SKIP_QWEN_NUMERIC_CHECK':'1',
                'FTLPU_SESSION_LINK_ONLY':'1','FTLPU_DDR_BANDWIDTH_MBPS':'7',
                'FTLPU_QWEN_STOP_AFTER_RMS2':'1','HF_TOKEN':'test credential'}):
            env = entry.clean_environment(self.root/'temp')
        self.assertNotIn('FTLPU_SKIP_QWEN_NUMERIC_CHECK',env)
        self.assertNotIn('FTLPU_SESSION_LINK_ONLY',env)
        self.assertNotIn('FTLPU_DDR_BANDWIDTH_MBPS',env)
        self.assertNotIn('FTLPU_QWEN_STOP_AFTER_RMS2',env)
        self.assertEqual(env['HF_TOKEN'],'test credential')

    def test_doctor_rejects_target_different_from_runtime_build(self):
        self.args.build_dir.mkdir()
        cached = self.root/'cached.json'
        cached.write_text('{"clock":750}')
        (self.args.build_dir/'CMakeCache.txt').write_text('FTLPU_HARDWARE_CONFIG:FILEPATH='+str(cached))
        issues = entry.check_environment(self.args,['qwen2_5_1_5b'],check_binaries=False)
        self.assertTrue(any('differs' in issue for issue in issues))

    def test_official_validation_rejects_changed_config_and_wrong_publisher(self):
        checkpoint = entry.checkpoint_directory(self.args,'qwen3_5_0_8b')
        checkpoint.mkdir(parents=True)
        (checkpoint/'config.json').write_text('{}')
        (checkpoint/'model.safetensors').write_bytes(b'raw official tensor subset')
        provenance = dict(repo='Qwen/Qwen3.5-0.8B',layer=0,embedding_token_ids=list(range(32)),
            subset_sha256=entry.digest(checkpoint/'model.safetensors'),
            config_sha256=entry.digest(checkpoint/'config.json'))
        (checkpoint/'provenance.json').write_text(json.dumps(provenance))
        self.assertEqual(entry.validate_checkpoint(self.args,'qwen3_5_0_8b')['repo'],provenance['repo'])
        (checkpoint/'config.json').write_text('{"hidden_size":64}')
        with self.assertRaisesRegex(ValueError,'mismatch'):
            entry.validate_checkpoint(self.args,'qwen3_5_0_8b')
        (checkpoint/'config.json').write_text('{}')
        provenance['repo'] = 'someone/synthetic-model'
        (checkpoint/'provenance.json').write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError,'mismatch'):
            entry.validate_checkpoint(self.args,'qwen3_5_0_8b')

    def test_real_subprocess_preserves_arguments_and_failure_code(self):
        log = self.root/'subprocess.log'
        argument = 'literal spaces & semicolon; $(must not execute)'
        command = [sys.executable,'-B','-c',
            'import sys; print(sys.argv[1]); raise SystemExit(7)',argument]
        with contextlib.redirect_stdout(io.StringIO()):
            code = entry.execute(command,log,entry.clean_environment(self.root))
        self.assertEqual(code,7)
        self.assertEqual(log.read_text().strip(),argument)

    def run_with_fake_backend(self, phases, execute):
        with patch.object(entry,'commands',return_value=phases), patch.object(entry,'execute',side_effect=execute), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return entry.run_case(self.args,'llama3_2_1b')

    def test_compile_only_never_claims_numerical_success(self):
        self.args.suite = 'compile'
        report = self.run_with_fake_backend([('compile',['fake-compiler'])],lambda *args:0)
        self.assertEqual(report['status'],'compiled')
        self.assertFalse(report['numeric_passed'])
        self.assertFalse(report['runtime_executed'])

    def test_failed_backend_cannot_reuse_previous_success(self):
        out = entry.result_directory(self.args,'llama3_2_1b')
        out.mkdir(parents=True)
        (out/'summary.json').write_text('{"status":"passed"}')
        (out/'icu_programs').mkdir()
        (out/'icu_programs/old.icu.csv').write_text('stale instructions')
        def fail(command, log, env):
            log.write_text('numerical mismatch')
            return 9
        report = self.run_with_fake_backend([('cmodel',['fake-runtime'])],fail)
        self.assertEqual(report['status'],'failed')
        self.assertFalse(report['numeric_passed'])
        self.assertTrue(report['runtime_attempted'])
        self.assertFalse((out/'summary.json').exists())
        self.assertFalse((out/'icu_programs/old.icu.csv').exists())
        self.assertEqual(json.loads((out/'runner.json').read_text())['status'],'failed')

    def test_gated_skip_requires_explicit_opt_in_and_real_gated_error(self):
        self.args.suite = 'official'
        def gated(command, log, env):
            log.write_text('HTTP 401 GatedRepo')
            return 1
        phases = [('download',['fake-download'])]
        self.assertEqual(self.run_with_fake_backend(phases,gated)['status'],'failed')
        self.args.allow_gated_skip = True
        report = self.run_with_fake_backend(phases,gated)
        self.assertEqual(report['status'],'skipped_access')
        self.assertFalse(report['numeric_passed'])
        def unrelated(command, log, env):
            log.write_text('connection timeout')
            return 1
        self.assertEqual(self.run_with_fake_backend(phases,unrelated)['status'],'failed')

    def test_dry_run_never_writes_downloads_or_executes(self):
        self.args.dry_run = True
        self.args.suite = 'official'
        self.args.download = True
        with patch.object(entry,'execute') as execute, contextlib.redirect_stdout(io.StringIO()):
            report = entry.run_case(self.args,'llama3_2_1b')
        execute.assert_not_called()
        self.assertEqual(report['status'],'planned')
        self.assertFalse(self.args.output_root.exists())
        self.assertFalse(self.args.checkpoint_root.exists())

    def test_requested_missing_pipeline_is_failure(self):
        self.args.no_pipeline = False
        report = self.run_with_fake_backend([('cmodel',['fake-runtime'])],lambda *args:0)
        self.assertEqual(report['status'],'failed')
        self.assertTrue(report['numeric_passed'])  # Preserve backend success, but fail the requested artifact.
        self.assertIn('pipeline CSV is missing',report['error'])


if __name__ == '__main__':
    unittest.main()
