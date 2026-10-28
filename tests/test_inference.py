"""Small CPU correctness checks; no network, pretrained weights or relaxation."""

import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models import CliqueFlowmer
from models.tools import batched_beam_search
from serving import inference


class Cancelled(Exception):
    pass


class QuadraticPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.grad_enabled = []

    def forward(self, x):
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.scale * x.square().sum(dim=(-2, -1))


class ObjectiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target_regressor = QuadraticPredictor()
        self.latent_dim = 3
        self.index_matrix = torch.tensor([[0, 1], [1, 2]])


class BeamModel:
    """A normalized autoregressive model with a known preferred stop length."""

    def __init__(self, stop_at=3, prefer_bos=False):
        self.stop_at = stop_at
        self.prefer_bos = prefer_bos
        self.calls = 0

    def atomic_emb(self, tokens):
        return tokens.unsqueeze(-1).float()

    def decoder(self, z, embedded, mask):
        self.calls += 1
        batch, length = embedded.shape[:2]
        logits = torch.full((batch, length, 120), -60.0)
        logits[..., 1] = 0.0
        logits[..., 2] = -0.5
        logits[..., 119] = 1.5 if length - 1 >= self.stop_at else -12.0
        if self.prefer_bos:
            logits[..., 0] = 5.0
        return logits.log_softmax(-1)


class DecodedModel(ObjectiveModel):
    def decode(self, z, **kwargs):
        self.decode_options = kwargs
        return (
            torch.full((1, 3), 4.0),
            torch.full((1, 3), torch.pi / 2),
            torch.tensor([[0, 14, 14, 119]]),
            torch.tensor([[[0., 0., 0.], [0., 0., 0.], [.25, .25, .25], [0., 0., 0.]]]),
            torch.ones(1, 4),
        )


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_es_preserves_input_minimizes_and_records_detached_values(self):
        torch.manual_seed(5)
        model = ObjectiveModel()
        z = torch.tensor([1.0, -0.8, 0.6], requires_grad=True)
        original = z.detach().clone()
        progress = []
        result, score, history = inference.optimize_es(
            model, z, steps=35, return_history=True,
            progress_callback=lambda step, total: progress.append((step, total)),
        )
        self.assertTrue(torch.equal(z.detach(), original))
        self.assertFalse(result.requires_grad)
        self.assertFalse(score.requires_grad)
        self.assertLess(history[-1]["objective"], history[0]["objective"])
        self.assertEqual([entry["step"] for entry in history], [0, 10, 20, 30, 35])
        self.assertEqual(progress, [(0, 35), (10, 35), (20, 35), (30, 35), (35, 35)])
        self.assertFalse(any(model.target_regressor.grad_enabled))
        self.assertTrue(all(param.grad is None for param in model.parameters()))
        self.assertEqual(len(inference.optimize_es(model, z, steps=0)), 2)

    def test_es_cancellation_does_not_modify_the_source(self):
        model = ObjectiveModel()
        z = torch.ones(3)
        calls = []

        def cancel_check():
            calls.append(True)
            if len(calls) == 2:
                raise Cancelled()

        with self.assertRaises(Cancelled):
            inference.optimize_es(model, z, steps=100, cancel_check=cancel_check)
        self.assertTrue(torch.equal(z, torch.ones(3)))
        self.assertEqual(len(calls), 2)

    def test_es_rejects_nonfinite_and_wrong_dimension(self):
        for z in (torch.ones(2), torch.tensor([1.0, float("nan"), 0.0])):
            with self.assertRaises(ValueError):
                inference.optimize_es(ObjectiveModel(), z)

    def test_beam_early_stop_matches_exhaustive_beam_search(self):
        for stop_at in (1, 3, 5):
            for beam_width in (1, 3, 10):
                with self.subTest(stop_at=stop_at, beam_width=beam_width):
                    full = BeamModel(stop_at)
                    stopped = BeamModel(stop_at)
                    kwargs = dict(beam_width=beam_width, min_atoms=1, max_atoms=12)
                    expected = batched_beam_search(full, torch.zeros(2, 4), early_stop=False, **kwargs)
                    actual = batched_beam_search(stopped, torch.zeros(2, 4), **kwargs)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
                    self.assertLess(stopped.calls, full.calls)

    def test_atom_limits_include_maximum_and_never_generate_bos(self):
        for limit in (1, 2, 5):
            result = batched_beam_search(
                BeamModel(stop_at=100, prefer_bos=True), torch.zeros(1, 4),
                min_atoms=limit, max_atoms=limit, beam_width=2,
            )[0]
            self.assertEqual(result.numel() - 2, limit)
            self.assertEqual(result[0].item(), 0)
            self.assertEqual(result[-1].item(), 119)
            self.assertTrue(((result[1:-1] >= 1) & (result[1:-1] <= 118)).all())

    def test_beam_cancels_during_search(self):
        model = BeamModel(stop_at=10)

        def cancel_check():
            if model.calls >= 2:
                raise Cancelled()

        with self.assertRaises(Cancelled):
            batched_beam_search(model, torch.zeros(1, 4), cancel_check=cancel_check)
        self.assertEqual(model.calls, 2)

    @staticmethod
    def tiny_model_kwargs():
        return dict(n_cliques=2, clique_dim=3, knot_dim=1, transformer_dim=16,
                    n_blocks=1, n_heads=2, n_registers=0, mlp_dim=8,
                    n_mlp=2, dropout_rate=0.0)

    def test_real_decode_hooks_preserve_seeded_results(self):
        model = CliqueFlowmer(**self.tiny_model_kwargs()).eval()
        z = torch.zeros(1, model.latent_dim)
        kwargs = dict(n_steps=12, beam_width=2, min_atoms=1, max_atoms=3)
        torch.manual_seed(7)
        expected = model.decode(z, **kwargs)
        progress = []
        torch.manual_seed(7)
        actual = model.decode(z, progress_callback=lambda step, total: progress.append((step, total)), **kwargs)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        self.assertEqual(progress[-1], (16, 16))
        self.assertEqual([step for step, _ in progress], sorted(step for step, _ in progress))
        self.assertTrue(all(1 <= int(count) - 2 <= 3 for count in actual[-1].sum(-1)))

    def test_cancellation_reaches_flow_loop(self):
        model = CliqueFlowmer(**self.tiny_model_kwargs()).eval()
        progress = []

        def cancel_check():
            if progress and progress[-1][0] >= 14:
                raise Cancelled()

        with self.assertRaises(Cancelled):
            model.decode(
                torch.zeros(1, model.latent_dim), n_steps=30, min_atoms=1, max_atoms=3,
                beam_width=1, cancel_check=cancel_check,
                progress_callback=lambda step, total: progress.append((step, total)),
            )
        self.assertLess(progress[-1][0], progress[-1][1])

    def test_weights_only_checkpoint_roundtrip(self):
        kwargs = self.tiny_model_kwargs()
        model = inference.build_model(kwargs, "cpu")
        state = {"module." + name: value for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save({"state_dict": state}, path)
            loaded = inference.load_model_from_checkpoint(path, kwargs, "cpu")
        self.assertFalse(loaded.training)
        self.assertTrue(all(torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items()))

    def test_hf_revision_is_used_for_both_artifacts_and_kwargs_are_unchanged(self):
        defaults = copy.deepcopy(inference.DEFAULT_MODEL_KWARGS)
        supplied = {"n_cliques": 2}
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / "prior.pickle"
            prior.write_bytes(pickle.dumps({"a": (1.0, 0.2)}))
            with patch.object(inference, "hf_hub_download", return_value=str(prior)) as download, \
                    patch.object(inference.data_tools, "normal_lengths_from_mle", return_value="prior"), \
                    patch.object(inference, "load_model_from_checkpoint", return_value="model") as load:
                result = inference.load_model_from_hf(
                    "approved/repo", "model.pt", model_kwargs=supplied, device="cpu", revision="abc123",
                )
        self.assertEqual(result, "model")
        self.assertEqual(supplied, {"n_cliques": 2})
        self.assertEqual(inference.DEFAULT_MODEL_KWARGS, defaults)
        self.assertTrue(all(call.kwargs["revision"] == "abc123" for call in download.call_args_list))
        self.assertEqual(download.call_count, 2)
        self.assertEqual(load.call_args.kwargs["model_kwargs"]["initial_length_dist"], "prior")

    def test_decode_rejects_nonfinite_latent_before_running_model(self):
        with self.assertRaises(ValueError):
            inference.decode_latent(ObjectiveModel(), np.array([0.0, np.nan, 1.0]), relax=False)

    def test_decode_forwards_limits_and_returns_actual_relaxed_result(self):
        model = DecodedModel()
        with patch.object(inference.data_tools, "refine_to_primitive_fast_strong") as relax:
            structure = inference.decode_latent(
                model, torch.zeros(3), relax=False, n_steps=42,
                beam_width=2, min_atoms=1, max_atoms=20,
            )
            relax.assert_not_called()
            self.assertEqual(len(structure), 2)
            self.assertEqual(model.decode_options["n_steps"], 42)
            self.assertEqual(model.decode_options["max_atoms"], 20)
            self.assertEqual(model.decode_options["min_atoms"], 1)
            relaxed = structure.copy()
            relaxed.scale_lattice(125)
            relax.return_value = relaxed
            result = inference.decode_latent(model, torch.zeros(3), relax=True)
            self.assertIs(result, relaxed)


if __name__ == "__main__":
    unittest.main()
