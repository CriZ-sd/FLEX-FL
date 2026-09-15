"""Dependency-free checks; extracted methods do not exercise PyTorch itself."""
import argparse
import ast
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shlex
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def tree(filename):
    return ast.parse((ROOT / filename).read_text(encoding='utf-8'))


def method(filename, class_name, name, **globals_):
    cls = next(n for n in tree(filename).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


def module(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, ROOT / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class RegressionTests(unittest.TestCase):
    def parse_runner(self, *arguments):
        main = next(n for n in tree('run_p2p_async_fl.py').body if isinstance(n, ast.If))
        stop = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'num_clients')
        ns = {'argparse': argparse, 'math': math}
        with patch.object(sys, 'argv', ['runner', '--stde_mode', '0', *arguments]):
            exec(compile(ast.Module(body=main.body[:stop], type_ignores=[]), 'runner', 'exec'), ns)
        return ns['args']

    def test_defaults_and_invalid_arguments(self):
        self.assertEqual(self.parse_runner().train_val, 'Classic')
        for args in [('--train_val', 'Classical'), ('--local_iters', '0'),
                     ('--total_iters', '-1'), ('--save_update_interval', '0'),
                     ('--n_peers', '10'), ('--n_peers', '0'), ('--adm_prob', '1.1'),
                     ('--latency_mu', 'nan'), ('--stde_mode', '-1'), ('--milestones', 'oops')]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse_runner(*args)

    def test_saved_command_round_trip(self):
        main = next(n for n in tree('run_p2p_async_fl.py').body if isinstance(n, ast.If))
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'command_args')
        end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'command')
        for sync in [False, True]:
            args = self.parse_runner('--data_root', 'data with spaces', *(['--sync'] if sync else []))
            ns = dict(args=args, shlex=shlex, milestones=[50, 75], gamma=0.1, weight_decay=0.0005)
            exec(compile(ast.Module(body=main.body[start:end+1], type_ignores=[]), 'command', 'exec'), ns)
            parsed = self.parse_runner(*shlex.split(ns['command'])[2:])
            self.assertEqual(parsed.sync, sync)
            self.assertEqual(parsed.data_root, 'data with spaces')
            self.assertEqual(parsed.milestones, '50,75')

    def test_iid_boolean(self):
        splitter = module('create_data_splits.py')
        for tail, expected in [([], False), (['--iid'], True), (['--iid', 'True'], True), (['--iid', 'False'], False)]:
            with patch.object(sys, 'argv', ['split', *tail]):
                self.assertIs(splitter.parse_args().iid, expected)

    def test_merge_repeated_run_and_shared_users(self):
        merger = module('merge_testset_femnist.py')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, label in [('a', 1), ('b', 2)]:
                (root / (name + '.json')).write_text(json.dumps({'user_data': {'writer': {'x': [[label]], 'y': [label]}}}))
            output = root / 'merged.json'
            with contextlib.redirect_stdout(io.StringIO()):
                merger.merge_femnist_json(root, output)
                first = output.read_bytes()
                merger.merge_femnist_json(root, output)
            self.assertEqual(output.read_bytes(), first)
            data = json.loads(first)
            self.assertEqual(data['users'], ['writer'])
            self.assertEqual(data['num_samples'], [2])

    def test_peers_exclude_self(self):
        add = method('client.py', 'Client', 'add_other_clients')
        clients = [types.SimpleNamespace(id=i) for i in [1, 2, 3]]
        local = types.SimpleNamespace(id=2, n_peers=2)
        add(local, clients)
        add(local, clients)
        self.assertEqual(local.client_ids, [1, 3])

    def test_reverse_aggregation_contract(self):
        aggregate = method('aggregation_algorithms.py', 'Aggregation_algorithm', 'aggregate')
        local = types.SimpleNamespace(fusion_protocol_name='peer_grad_ada_rev',
                                      peer_grad_ada_rev=lambda updates: ({'w': 2}, [0.5]))
        self.assertEqual(aggregate(local, [], 1, 0), ({'w': 2}, [0.5], 0))

    def test_adaptive_numerical_edges(self):
        nodes = list(ast.walk(tree('aggregation_algorithms.py')))
        scaled = [n.value for n in nodes if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'scaled_diff']
        self.assertEqual(len(scaled), 3)
        for expression in scaled:
            compiled = compile(ast.Expression(expression), 'scaled_diff', 'eval')
            self.assertEqual(eval(compiled, dict(p=0, base_progress=0, diff=5)), 0)
            self.assertEqual(eval(compiled, dict(p=1, base_progress=1, diff=5)), 2.5)
        angle = next(n for n in nodes if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'arccos')
        np = types.SimpleNamespace(arccos=math.acos, clip=lambda x, lo, hi: min(hi, max(lo, x)))
        compiled = compile(ast.Expression(angle), 'angle', 'eval')
        self.assertEqual(eval(compiled, dict(np=np, cos_theta=1.0000001)), 0)
        self.assertEqual(eval(compiled, dict(np=np, cos_theta=-1.0000001)), math.pi)

    def test_training_stops_at_total_for_both_methods(self):
        class Value:
            def to(self, device): return self
            def backward(self): pass
            def __iadd__(self, other): return self
        for name in ['local_train_valid_classic', 'local_train_valid_fedprox']:
            train = method('client.py', 'Client', name, time=time)
            local = types.SimpleNamespace(clientLogger=Mock(), local_iters=10, total_iters=5,
                current_iter=3, current_iter_sync=3, latency=0, sync_lock=threading.Condition(),
                gpu='cpu', model=Mock(return_value=Value()), optimizer=Mock(), scheduler=Mock(),
                loss=lambda *args: Value(), valid_interval=100, update_experimental=Mock(),
                prev_model=Mock())
            local.prev_model.state_dict.return_value = {}
            local.model.named_parameters.return_value = []
            train(local, [(Value(), Value())], [])
            self.assertEqual(local.current_iter, 5)
            self.assertEqual(local.optimizer.step.call_count, 2)

    def test_checkpoint_interval_and_final(self):
        start = method('client.py', 'Client', 'start_client', time=time)
        for total in [20, 25]:
            saved = []
            local = types.SimpleNamespace(current_iter=0, total_iters=total, local_iters=10,
                clientLogger=Mock(), experimental={}, model=Mock(), create_work_dir=Mock(),
                save_update_interval=10, log_update_interval=10, update_experimental=Mock())
            local.local_round = lambda: setattr(local, 'current_iter', min(total, local.current_iter + 10))
            local.clientLogger.save_update_log.side_effect = lambda model: saved.append(local.current_iter)
            start(local)
            self.assertEqual(saved, [10, 20] if total == 20 else [10, 20, 25])


if __name__ == '__main__':
    unittest.main()
