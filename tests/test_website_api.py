"""HTTP boundaries for the custom website, without model downloads."""
import time
import json
import unittest

from starlette.testclient import TestClient

from serving.web_demo.service import EXAMPLES, Job
from serving.website.api import BODY_LIMIT, COOKIE_NAME, create_app

CIF = EXAMPLES["Strontium titanate · 5 sites"]
ORIGIN = "https://demo.example"


class Worker:
    def __init__(self):
        self.jobs = []
        self.busy = False

    def submit(self, cif, **settings):
        if self.busy:
            raise ValueError("The CPU worker is busy")
        job = Job()
        self.jobs.append(job)
        return job

    def close(self):
        for job in self.jobs:
            job.cancel()


class WebsiteAPI(unittest.TestCase):
    def setUp(self):
        self.worker = Worker()
        self.app = create_app(worker=self.worker, public_origin=ORIGIN,
                              proxy_key="test-only-key")
        self.client = TestClient(self.app, base_url=ORIGIN,
                                 headers={"Origin": ORIGIN, "x-cliqueflowmer-key": "test-only-key"})
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def submit(self):
        response = self.client.post("/api/jobs", json={"cif": CIF})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["job_id"]

    def test_api_key_and_origin_boundaries(self):
        outsider = TestClient(self.app, base_url=ORIGIN)
        self.assertEqual(outsider.get("/api/config").status_code, 403)
        self.assertEqual(self.client.post("/api/jobs", json={"cif": CIF},
                         headers={"Origin": "https://unrelated.example"}).status_code, 403)
        self.assertEqual(self.worker.jobs, [])

    def test_preview_and_secure_session(self):
        response = self.client.post("/api/preview", json={"cif": CIF})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"]["n_sites"], 5)
        self.assertEqual(len(response.json()["viewer"]["atoms"]), 5)
        cookie = response.headers["set-cookie"].lower()
        for flag in ("httponly", "secure", "samesite=strict"):
            self.assertIn(flag, cookie)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_job_is_private_to_its_signed_session(self):
        job_id = self.submit()
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").status_code, 200)
        other = TestClient(self.app, base_url=ORIGIN,
                           headers={"Origin": ORIGIN, "x-cliqueflowmer-key": "test-only-key"})
        self.assertEqual(other.get(f"/api/jobs/{job_id}").status_code, 404)
        self.assertEqual(other.post(f"/api/jobs/{job_id}/cancel", json={}).status_code, 404)
        self.assertFalse(self.worker.jobs[0]._cancel.is_set())
        cookie = self.client.cookies.get(COOKIE_NAME)
        self.client.cookies.clear()
        self.client.cookies.set(COOKIE_NAME, cookie[:-1] + ("0" if cookie[-1] != "0" else "1"))
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").status_code, 404)

    def test_cancel_replacement_and_busy(self):
        old_id = self.submit()
        self.worker.busy = True
        response = self.client.post("/api/jobs", json={"cif": CIF})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "5")
        self.assertEqual(self.client.get(f"/api/jobs/{old_id}").status_code, 200)
        self.worker.busy = False
        new_id = self.submit()
        self.assertTrue(self.worker.jobs[0]._cancel.is_set())
        self.assertEqual(self.client.get(f"/api/jobs/{old_id}").status_code, 404)
        response = self.client.post(f"/api/jobs/{new_id}/cancel", json={})
        self.assertEqual(response.json()["status"], "cancelling")
        self.assertTrue(self.worker.jobs[-1]._cancel.is_set())

    def test_invalid_input_never_runs_a_job(self):
        for change in ({"seed": True}, {"seed": -1}, {"seed": 2**31},
                       {"operation": []}, {"budget": {}}, {"cif": "not a CIF"},
                       {"cif": "x" * 100001}, {"cif": "\ud800"}):
            with self.subTest(change=list(change)):
                self.assertEqual(self.client.post("/api/jobs", content=json.dumps({"cif": CIF, **change}), headers={"Content-Type": "application/json"}).status_code, 422)
        self.assertEqual(self.client.post("/api/preview", content=b"x" * (BODY_LIMIT + 1),
                         headers={"Content-Type": "application/json"}).status_code, 413)
        self.assertEqual(self.client.post("/api/preview", content="{}").status_code, 415)
        self.assertEqual(self.client.post("/api/preview", json=[]).status_code, 422)
        self.assertEqual(self.worker.jobs, [])

    def test_result_enrichment_and_expiration(self):
        job_id = self.submit()
        job = self.worker.jobs[0]
        job.update(status="completed", result={"input_cif": CIF, "output_cif": CIF})
        result = self.client.get(f"/api/jobs/{job_id}").json()["result"]
        self.assertEqual(len(result["output_viewer"]["atoms"]), 5)
        self.assertNotIn("output_viewer", job.snapshot()["result"])
        job._finished = time.monotonic() - 1801
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").status_code, 404)

    def test_registry_is_bounded(self):
        self.app.state.registry.limit = 1
        self.submit()
        self.client.cookies.clear()
        self.assertEqual(self.client.post("/api/jobs", json={"cif": CIF}).status_code, 429)
        self.assertEqual(len(self.worker.jobs), 1)


if __name__ == "__main__":
    unittest.main()
