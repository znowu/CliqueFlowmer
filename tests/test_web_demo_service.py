"""Public-input and worker lifecycle checks without downloading a checkpoint."""

import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from serving.web_demo import service


VALID_CIF = service.EXAMPLES["Silicon · 2 sites"]


def cif_with(rows, operations=("x,y,z",), length=10):
    return f"""data_test
_cell_length_a {length}
_cell_length_b {length}
_cell_length_c {length}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_symmetry_equiv_pos_as_xyz
""" + "\n".join(repr(op) for op in operations) + """
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
""" + "\n".join(rows) + "\n"


class ValidationTests(unittest.TestCase):
    def test_bundled_examples_parse_and_preserve_uploaded_cell(self):
        for name, expected_sites in [("Silicon · 2 sites", 2), ("Copper · 4 sites", 4),
                                     ("Strontium titanate · 5 sites", 5)]:
            with self.subTest(example=name):
                structure = service.validate_cif(service.EXAMPLES[name])
                self.assertEqual(len(structure), expected_sites)
                self.assertTrue(structure.is_ordered)

    def test_rejects_multiple_blocks_even_with_identical_headers(self):
        for second in (VALID_CIF, VALID_CIF.replace("data_Si", "data_other")):
            with self.subTest(same_header=second == VALID_CIF):
                with self.assertRaisesRegex(ValueError, "exactly one CIF data block"):
                    service.validate_cif(VALID_CIF + "\n" + second)

    def test_rejects_empty_invalid_and_oversized_text_before_parsing(self):
        for value in (None, "", " \n\t"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "CIF first"):
                service.validate_cif(value)
        # Limits are UTF-8 bytes, not character counts.
        oversized = VALID_CIF + "\n# " + "é" * (service.MAX_CIF_BYTES // 2)
        with patch.object(service.CifParser, "from_str") as parser:
            with self.assertRaisesRegex(ValueError, "100 KB"):
                service.validate_cif(oversized)
            parser.assert_not_called()
        with self.assertRaises(ValueError):
            service.validate_cif("data_broken\n_cell_length_a not_a_number\n")

    def test_rejects_partial_occupancy(self):
        with self.assertRaisesRegex(ValueError, "fully occupied"):
            service.validate_cif(cif_with(["Si Si0 0 0 0 0.5"]))

    def test_bounds_raw_atom_rows_before_symmetry_expansion(self):
        text = cif_with([f"Si Si{i} {i / 21} 0 0 1" for i in range(21)])
        with patch.object(service.CifParser, "from_str") as parser:
            with self.assertRaisesRegex(ValueError, "20 atom-site rows"):
                service.validate_cif(text)
            parser.assert_not_called()

    def test_bounds_symmetry_operations_before_structure_parser(self):
        text = cif_with(["Si Si0 0 0 0 1"], operations=["x,y,z"] * 193)
        with patch.object(service.CifParser, "from_str") as parser:
            with self.assertRaisesRegex(ValueError, "too many symmetry"):
                service.validate_cif(text)
            parser.assert_not_called()

    def test_checks_site_count_after_symmetry_expansion(self):
        text = cif_with(["Si Si0 0 0 0 1"], operations=[f"x+{i}/21,y,z" for i in range(21)])
        with self.assertRaisesRegex(ValueError, "1–20 sites"):
            service.validate_cif(text)

    def test_rejects_extreme_cells_and_nonfinite_coordinates(self):
        for length in (0.01, 1000):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "finite, nondegenerate"):
                service.validate_cif(cif_with(["Si Si0 0 0 0 1"], length=length))
        structure = Structure(Lattice.cubic(5), ["Si"], [[np.nan, 0, 0]])
        with self.assertRaisesRegex(ValueError, "finite, nondegenerate"):
            service.validate_structure(structure)

    def test_rejects_overlapping_sites_and_dummy_species(self):
        with self.assertRaisesRegex(ValueError, "less than 0.5"):
            service.validate_cif(cif_with(["Si Si0 0 0 0 1", "O O0 0.01 0 0 1"]))
        with self.assertRaisesRegex(ValueError, "supported chemical element"):
            service.validate_structure(Structure(Lattice.cubic(5), ["X"], [[0, 0, 0]]))


class JobTests(unittest.TestCase):
    def test_snapshots_are_deeply_isolated(self):
        job = service.Job()
        job.update(status="completed", result={"latent": [1., 2.], "metadata": {"seed": 42}})
        snapshot = job.snapshot()
        snapshot["result"]["latent"][0] = 999
        snapshot["result"]["metadata"]["seed"] = -1
        snapshot["status"] = "failed"
        fresh = job.snapshot()
        self.assertEqual(fresh["status"], "completed")
        self.assertEqual(fresh["result"], {"latent": [1., 2.], "metadata": {"seed": 42}})
        self.assertIsNone(service.Job().snapshot()["result"])

    def test_results_expire_on_access_even_when_finished_at_zero(self):
        job = service.Job()
        with patch.object(service.time, "monotonic", return_value=0):
            job.update(status="completed", result={"private": "uploaded structure"})
        with patch.object(service.time, "monotonic", return_value=service.RESULT_TTL_SECONDS - 1):
            self.assertIsNotNone(job.snapshot()["result"])
        with patch.object(service.time, "monotonic", return_value=service.RESULT_TTL_SECONDS + 1):
            expired = job.snapshot()
        self.assertIsNone(expired["result"])
        self.assertIsNone(job._state["result"])
        self.assertEqual(expired["status"], "failed")
        self.assertIn("expired", expired["error"])
        self.assertIn("expired", expired["message"])

    def test_unfinished_job_has_no_expiry(self):
        job = service.Job()
        with patch.object(service.time, "monotonic", return_value=10**9):
            self.assertEqual(job.snapshot()["status"], "queued")


class WorkerTests(unittest.TestCase):
    def make_service(self, runner, model_loader=None, **kwargs):
        worker = service.DemoService(model_loader=model_loader or Mock(return_value=object()),
                                     runner=runner, **kwargs)
        self.addCleanup(worker.close)
        return worker

    def completed(self, worker, job):
        # Wait behind the worker to include its finally block and slot release.
        worker._executor.submit(lambda: None).result(timeout=5)
        return job.snapshot()

    def test_busy_rejection_and_release_after_cancellation(self):
        entered = threading.Event()
        release = threading.Event()

        def run(model, text, *, cancel_check, **kwargs):
            entered.set()
            if not release.wait(3):
                raise AssertionError("Test runner was not released")
            cancel_check()
            return {"seed": kwargs["seed"]}

        worker = self.make_service(run)
        self.addCleanup(release.set)
        first = worker.submit(VALID_CIF)
        self.assertTrue(entered.wait(3))
        with self.assertRaisesRegex(ValueError, "busy"):
            worker.submit(VALID_CIF)
        first.cancel()
        release.set()
        self.assertEqual(self.completed(worker, first)["status"], "cancelled")
        second = worker.submit(VALID_CIF, seed=7)
        self.assertEqual(self.completed(worker, second)["result"], {"seed": 7})

    def test_cancel_during_initial_load_prevents_runner_execution(self):
        entered = threading.Event()
        release = threading.Event()

        def load():
            entered.set()
            if not release.wait(3):
                raise AssertionError("Test loader was not released")
            return object()

        runner = Mock(return_value={})
        worker = self.make_service(runner, model_loader=load)
        self.addCleanup(release.set)
        job = worker.submit(VALID_CIF)
        self.assertTrue(entered.wait(3))
        job.cancel()
        release.set()
        self.assertEqual(self.completed(worker, job)["status"], "cancelled")
        runner.assert_not_called()

    def test_timeout_discards_results_and_releases_slot(self):
        with patch.object(service.time, "monotonic", return_value=0) as clock:
            def run(model, text, *, cancel_check, **kwargs):
                clock.return_value = 2
                cancel_check()
                return {"should_not_be_visible": True}

            worker = self.make_service(run, timeout_seconds=1)
            job = worker.submit(VALID_CIF)
            state = self.completed(worker, job)
            self.assertEqual(state["status"], "failed")
            self.assertIn("time limit", state["error"])
            self.assertIsNone(state["result"])
            worker._runner = Mock(return_value={"ok": True})
            self.assertEqual(self.completed(worker, worker.submit(VALID_CIF))["status"], "completed")

    def test_runner_errors_release_worker_and_do_not_expose_internal_exception(self):
        runner = Mock(side_effect=[RuntimeError("secret internal diagnostic"), {"ok": True}])
        worker = self.make_service(runner)
        with self.assertLogs(service.log, level="ERROR"):
            state = self.completed(worker, worker.submit(VALID_CIF))
        self.assertEqual(state["status"], "failed")
        self.assertNotIn("secret", state["error"])
        self.assertIsNone(state["result"])
        self.assertEqual(self.completed(worker, worker.submit(VALID_CIF))["result"], {"ok": True})

    def test_invalid_input_never_loads_the_model_and_releases_slot(self):
        loader = Mock(return_value=object())
        worker = self.make_service(Mock(return_value={"ok": True}), model_loader=loader)
        state = self.completed(worker, worker.submit("data_bad\n"))
        self.assertEqual(state["status"], "failed")
        loader.assert_not_called()
        self.assertEqual(self.completed(worker, worker.submit(VALID_CIF))["status"], "completed")
        loader.assert_called_once()

    def test_model_is_reused_without_sharing_job_results(self):
        loader = Mock(return_value=object())
        worker = self.make_service(lambda model, text, **kw: {"latent": [kw["seed"]]},
                                   model_loader=loader)
        first = worker.submit(VALID_CIF, seed=1)
        first_state = self.completed(worker, first)
        second = worker.submit(VALID_CIF, seed=2)
        second_state = self.completed(worker, second)
        first_state["result"]["latent"][0] = 1234
        self.assertEqual(first.snapshot()["result"], {"latent": [1]})
        self.assertEqual(second_state["result"], {"latent": [2]})
        loader.assert_called_once()

    def test_submission_failure_releases_the_reserved_slot(self):
        worker = self.make_service(Mock(return_value={"ok": True}))
        with patch.object(worker._executor, "submit", side_effect=RuntimeError("executor unavailable")):
            with self.assertRaisesRegex(RuntimeError, "executor unavailable"):
                worker.submit(VALID_CIF)
        self.assertEqual(self.completed(worker, worker.submit(VALID_CIF))["status"], "completed")

    def test_invalid_request_settings_do_not_reserve_worker(self):
        worker = self.make_service(Mock(return_value={"ok": True}))
        for settings in ({"operation": "unknown"}, {"budget": "unbounded"},
                         {"seed": -1}, {"seed": 2**31}, {"seed": 1.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                worker.submit(VALID_CIF, **settings)
        self.assertEqual(self.completed(worker, worker.submit(VALID_CIF))["status"], "completed")


class PipelineRetryTests(unittest.TestCase):
    """Exercise the seeded retry policy without real decoding or checkpoints."""

    def setUp(self):
        class SumPredictor(torch.nn.Module):
            def forward(self, separated):
                return separated.sum(dim=(-1, -2))

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.index_matrix = torch.tensor([[0, 1], [1, 2]])
                self.target_regressor = SumPredictor()

        self.model = TinyModel()
        self.source_latent = torch.tensor([1., 2., 3.])
        self.optimized_latent = torch.tensor([0.1, 0.2, 0.3])
        self.output = service.validate_cif(VALID_CIF)
        self.encode = self.enterContext(patch(
            "serving.inference.encode_structure", return_value=self.source_latent,
        ))
        self.optimize = self.enterContext(patch(
            "serving.inference.optimize_es",
            return_value=(self.optimized_latent, torch.tensor(0.8), [
                {"step": 0, "objective": 8.}, {"step": 100, "objective": 0.8},
            ]),
        ))
        self.decode = self.enterContext(patch("serving.inference.decode_latent"))

    def test_retry_keeps_optimized_latent_and_continues_seeded_rng(self):
        decoded_latents = []
        random_draws = []
        seed = 91

        def encode(model, structure):
            torch.rand(())  # Account for randomness consumed before decoding.
            return self.source_latent

        self.encode.side_effect = encode

        def decode(model, latent, **kwargs):
            decoded_latents.append(latent.detach().clone())
            random_draws.append(torch.rand(()).item())
            if len(random_draws) == 1:
                raise ValueError("No valid structure produced by decoder.")
            return self.output.copy()

        self.decode.side_effect = decode
        outside_rng = torch.random.get_rng_state().clone()
        result = service.run_pipeline(self.model, VALID_CIF, seed=seed)
        self.assertTrue(torch.equal(outside_rng, torch.random.get_rng_state()))
        expected_draws = torch.rand(3, generator=torch.Generator().manual_seed(seed)).tolist()[1:]
        self.assertEqual(random_draws, expected_draws)
        self.assertNotEqual(random_draws[0], random_draws[1])
        self.assertEqual(self.decode.call_count, 2)
        self.optimize.assert_called_once()
        self.assertTrue(all(torch.equal(value, self.optimized_latent) for value in decoded_latents))
        self.assertEqual(result["metadata"]["decode_attempts"], 2)
        self.assertTrue(torch.equal(self.source_latent, torch.tensor([1., 2., 3.])))
        self.assertEqual(result["latent"], self.source_latent.tolist())
        self.assertEqual(result["optimized_latent"], self.optimized_latent.tolist())
        input_structure = self.encode.call_args.args[1]
        self.assertEqual(input_structure, service.validate_cif(VALID_CIF))
        self.assertEqual(service.validate_cif(result["input_cif"]), input_structure)

        # A second run with the same seed repeats the full stream, including retry.
        random_draws.clear()
        decoded_latents.clear()
        again = service.run_pipeline(self.model, VALID_CIF, seed=seed)
        self.assertEqual(random_draws, expected_draws)
        self.assertEqual(again["output_cif"], result["output_cif"])
        self.assertEqual(again["optimized_latent"], result["optimized_latent"])

    def test_invalid_returned_geometry_gets_one_retry(self):
        invalid = Structure(Lattice.cubic(0.01), ["Si"], [[0, 0, 0]])
        self.decode.side_effect = [invalid, self.output.copy()]
        result = service.run_pipeline(self.model, VALID_CIF)
        self.assertEqual(self.decode.call_count, 2)
        self.assertEqual(result["metadata"]["decode_attempts"], 2)

    def test_geometric_failure_never_exceeds_two_attempts(self):
        self.decode.side_effect = ValueError("No valid structure produced by decoder.")
        with self.assertRaisesRegex(ValueError, "Two decoded geometries failed validation"):
            service.run_pipeline(self.model, VALID_CIF)
        self.assertEqual(self.decode.call_count, 2)
        self.optimize.assert_called_once()

    def test_cancellation_timeout_and_programming_errors_are_not_retried(self):
        for error in (service.Cancelled(), TimeoutError("deadline"),
                      RuntimeError("bad tensor operation"), ValueError("Unexpected latent shape")):
            with self.subTest(error=type(error).__name__):
                self.decode.reset_mock()
                self.decode.side_effect = error
                before = torch.random.get_rng_state().clone()
                with self.assertRaises(type(error)):
                    service.run_pipeline(self.model, VALID_CIF)
                self.assertEqual(self.decode.call_count, 1)
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_cancellation_between_attempts_prevents_retry(self):
        failed = threading.Event()

        def decode(*args, **kwargs):
            failed.set()
            raise ValueError("No valid structure produced by decoder.")

        def cancel_check():
            if failed.is_set():
                raise service.Cancelled()

        self.decode.side_effect = decode
        with self.assertRaises(service.Cancelled):
            service.run_pipeline(self.model, VALID_CIF, cancel_check=cancel_check)
        self.assertEqual(self.decode.call_count, 1)

    def test_attempt_metadata_distinguishes_first_success_and_encode_only(self):
        self.decode.return_value = self.output.copy()
        result = service.run_pipeline(self.model, VALID_CIF, operation="reconstruct")
        self.assertEqual(result["metadata"]["decode_attempts"], 1)
        self.assertIsNone(result["optimized_latent"])
        self.optimize.assert_not_called()
        self.decode.reset_mock()
        result = service.run_pipeline(self.model, VALID_CIF, operation="encode")
        self.decode.assert_not_called()
        self.assertEqual(result["metadata"]["decode_attempts"], 0)
        self.assertIsNone(result["output_cif"])


if __name__ == "__main__":
    unittest.main()
