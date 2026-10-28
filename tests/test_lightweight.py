"""Regression checks for extracting inference from the research dependency stack."""

import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch

from data.constants import atomic_numbers
from data import tensor_tools


ROOT = Path(__file__).resolve().parents[1]


def legacy_tensor_helpers():
    """Load just the original functions, without executing heavy module imports."""
    path = ROOT / "data" / "tools.py"
    names = {
        "move_to_device", "pad_sequences_fast", "_metric_from_lattice",
        "_min_image_exact", "compute_pairwise_distances", "causal_mask",
    }
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "torch": torch, "np": np, "deepcopy": deepcopy,
        "atomic_numbers": atomic_numbers,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TensorHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = legacy_tensor_helpers()

    def test_padding_matches_training_for_supported_sequence_types(self):
        sequences = [
            [[0, 14, 8, 119], [0, 26, 119]],
            [np.zeros((2, 3), dtype=np.float32), np.ones((3, 3), dtype=np.float32)],
            [[np.zeros(3)], [np.ones(3), np.full(3, 0.5)]],
            [torch.tensor([1., 2.]), torch.tensor([3.])],
        ]
        for value in sequences:
            with self.subTest(sequence_type=type(value[0])):
                expected = self.legacy["pad_sequences_fast"](value)
                actual = tensor_tools.pad_sequences_fast(value)
                for before, after in zip(expected, actual):
                    self.assertTrue(torch.equal(before, after))
                    self.assertEqual(before.dtype, after.dtype)

    def test_distances_match_training_in_orthogonal_and_triclinic_cells(self):
        generator = torch.Generator().manual_seed(42)
        abc = torch.tensor([[4., 5., 6.], [3.2, 7.1, 4.4]], dtype=torch.float64)
        angles = torch.deg2rad(torch.tensor([[90., 90., 90.], [78., 103., 61.]], dtype=torch.float64))
        positions = torch.rand((2, 5, 3), generator=generator, dtype=torch.float64)
        for periodic, exact in [(False, True), (True, False), (True, True)]:
            with self.subTest(periodic=periodic, exact=exact):
                expected = self.legacy["compute_pairwise_distances"](
                    abc, angles, positions, periodic=periodic, exact_min_image=exact,
                )
                actual = tensor_tools.compute_pairwise_distances(
                    abc, angles, positions, periodic=periodic, exact_min_image=exact,
                )
                self.assertTrue(torch.equal(expected, actual))

    def test_periodic_distance_crosses_unit_cell_boundary(self):
        abc = torch.tensor([[4., 4., 4.]])
        angles = torch.full((1, 3), torch.pi / 2)
        positions = torch.tensor([[[0.1, 0., 0.], [0.9, 0., 0.]]])
        periodic = tensor_tools.compute_pairwise_distances(abc, angles, positions)
        unwrapped = tensor_tools.compute_pairwise_distances(abc, angles, positions, periodic=False)
        self.assertAlmostEqual(periodic[0, 0, 1].item(), 0.8, places=5)
        self.assertAlmostEqual(unwrapped[0, 0, 1].item(), 3.2, places=5)

    def test_mask_and_device_helpers_match_training(self):
        for shape in [(3,), (2, 4), (2, 3, 5)]:
            value = torch.zeros(shape)
            self.assertTrue(torch.equal(
                tensor_tools.causal_mask(value), self.legacy["causal_mask"](value),
            ))
        tensors = [torch.tensor([1, 2]), torch.tensor([[0.5]])]
        actual = tensor_tools.move_to_device(tensors, "cpu")
        expected = self.legacy["move_to_device"](tensors, "cpu")
        self.assertIsInstance(actual, tuple)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))


class ImportBoundaryTests(unittest.TestCase):
    def test_core_model_and_serving_helpers_do_not_import_optional_stacks(self):
        script = """
import importlib.abc
import sys

class RejectOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'matgl', 'ase', 'wandb', 'matplotlib'}
                or fullname in {'google.cloud', 'data.tools', 'models.transformer', 'saving'}):
            raise RuntimeError('Unexpected optional import: ' + fullname)

sys.meta_path.insert(0, RejectOptionalImports())
import models
assert 'models.cliqueflowmer' not in sys.modules
import data.tensor_tools
import serving.data_tools
from models import CliqueFlowmer, Flow
assert 'serving.relaxation' not in sys.modules
assert 'models.transformer' not in sys.modules
assert CliqueFlowmer is models.CliqueFlowmer
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
