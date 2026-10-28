"""UI state regressions; no checkpoint download or model inference required."""

from copy import deepcopy
from pathlib import Path
import time
import unittest

from streamlit.testing.v1 import AppTest

from serving.web_demo import service


APP = Path(__file__).resolve().parents[1] / "serving/web_demo/app.py"


class StubJob:
    def __init__(self, state):
        self.state = state
        self.cancelled = False

    def snapshot(self):
        return deepcopy(self.state)

    def cancel(self):
        self.cancelled = True


class DemoUiTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertNoException()

    def assertNoException(self):
        self.assertEqual([error.message for error in self.app.exception], [])

    def complete_job(self):
        cif = next(iter(service.EXAMPLES.values()))
        summary = service.structure_summary(service.validate_cif(cif))
        result = {
            "input_cif": cif, "output_cif": cif,
            "input_summary": summary, "output_summary": summary,
            "latent": [index / 100 for index in range(121)],
            "optimized_latent": [index / 101 for index in range(121)],
            "history": [{"step": 0, "score": 1.5}, {"step": 100, "score": 1.2}],
            "initial_score": 1.5, "final_score": 1.2,
            "metadata": {"seed": 42, "property_name": "Latent objective", "unit": "model score"},
        }
        self.app.session_state["cf_operation"] = "optimize"
        self.app.session_state["cf_job"] = StubJob({"status": "completed", "result": result})
        self.app.run()
        self.assertNoException()
        return result

    def test_completion_retains_result_and_exposes_downloads(self):
        result = self.complete_job()
        self.assertEqual(self.app.session_state["cf_result"], result)
        self.assertIsNone(self.app.session_state["cf_job"])
        self.assertEqual(len(self.app.get("download_button")), 4)
        self.assertEqual(len(self.app.metric), 3)
        self.app.radio(key="cf_latent_view").set_value("Optimized").run()
        self.assertNoException()
        self.assertEqual(self.app.session_state["cf_result"], result)

    def test_invalid_input_clears_stale_result_and_disables_actions(self):
        self.complete_job()
        self.app.radio(key="cf_input_mode").set_value("Paste CIF").run()
        self.app.text_area(key="cf_paste").set_value("not a CIF").run()
        self.assertNoException()
        self.assertIsNone(self.app.session_state["cf_result"])
        self.assertIsNone(self.app.session_state["cf_preview"])
        self.assertIsNone(self.app.session_state["cf_output_preview"])
        self.assertTrue(self.app.warning)
        self.assertTrue(all(button.disabled for button in self.app.button))

    def test_input_change_cancels_and_detaches_previous_job(self):
        job = StubJob({"status": "running", "progress": 0.2, "message": "Optimizing"})
        self.app.session_state["cf_job"] = job
        self.app.run()
        self.app.radio(key="cf_input_mode").set_value("Paste CIF").run()
        self.assertNoException()
        self.assertTrue(job.cancelled)
        self.assertIsNone(self.app.session_state["cf_job"])
        self.assertIsNone(self.app.session_state["cf_result"])
        # A late completion of the detached job must not restore the old result.
        job.state.update(status="completed", result={"stale": True})
        self.app.run()
        self.assertIsNone(self.app.session_state["cf_result"])

    def test_cancel_remains_pending_until_worker_stops(self):
        job = StubJob({"status": "running", "progress": 0.2, "message": "Optimizing"})
        self.app.session_state["cf_job"] = job
        self.app.run()
        self.app.button(key="cf_cancel").click().run()
        self.assertNoException()
        self.assertTrue(job.cancelled)
        self.assertTrue(self.app.session_state["cf_cancel_requested"])
        self.assertIsNotNone(self.app.session_state["cf_job"])
        job.state.update(status="cancelled")
        self.app.run()
        self.assertIsNone(self.app.session_state["cf_job"])
        self.assertTrue(any("cancelled" in message.value for message in self.app.info))

    def test_completed_results_expire_from_ui_state(self):
        self.complete_job()
        self.app.session_state["cf_result_created"] = time.monotonic() - service.RESULT_TTL_SECONDS - 1
        self.app.run()
        self.assertNoException()
        self.assertIsNone(self.app.session_state["cf_result"])
        self.assertIsNone(self.app.session_state["cf_output_preview"])
        self.assertIsNone(self.app.session_state["cf_result_created"])
        self.assertTrue(any("expired" in message.value for message in self.app.info))

    def test_oversized_paste_is_rejected_without_structure_parsing(self):
        from unittest.mock import patch

        self.app.radio(key="cf_input_mode").set_value("Paste CIF").run()
        with patch.object(service, "validate_cif", side_effect=AssertionError("Should not parse oversized input")):
            self.app.text_area(key="cf_paste").set_value("é" * (service.MAX_CIF_BYTES // 2 + 1)).run()
        self.assertNoException()
        self.assertIsNone(self.app.session_state["cf_preview"])
        self.assertTrue(any("100 KB" in message.value for message in self.app.warning))


if __name__ == "__main__":
    unittest.main()
