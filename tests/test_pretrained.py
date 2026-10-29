"""Pinned artifact integrity and prior checks without network or model downloads."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from serving import pretrained


class PretrainedTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "checkpoint.pth"
        self.payload = b"verified checkpoint fixture"
        self.path.write_bytes(self.payload)
        self.manifest = dict(pretrained.MANIFEST, sha256=hashlib.sha256(self.payload).hexdigest())

    def test_download_uses_pinned_revision_and_matching_bundled_prior(self):
        expected_model = object()
        threads = torch.get_num_threads()
        with patch.dict(os.environ, {"CLIQUEFLOWMER_CHECKPOINT": ""}, clear=True), \
                patch.object(pretrained, "MANIFEST", self.manifest), \
                patch.object(pretrained, "hf_hub_download", return_value=str(self.path)) as download, \
                patch.object(pretrained, "load_model_from_checkpoint", return_value=expected_model) as load:
            model = pretrained.load_pretrained_model()

        self.assertIs(model, expected_model)
        download.assert_called_once_with(
            repo_id=self.manifest["repo_id"], filename=self.manifest["filename"],
            revision=self.manifest["revision"],
        )
        self.assertEqual(load.call_args.args, (self.path,))
        self.assertEqual(load.call_args.kwargs["device"], "cpu")
        prior_values = json.loads((pretrained.HERE / "length_prior.json").read_text())
        distribution = load.call_args.kwargs["model_kwargs"]["initial_length_dist"]
        means = torch.tensor([prior_values[key][0] for key in ("a", "b", "c")]).float()
        stds = torch.tensor([prior_values[key][1] for key in ("a", "b", "c")]).float()
        torch.testing.assert_close(distribution.loc, means)
        torch.testing.assert_close(distribution.covariance_matrix, torch.diag(stds.square()))
        self.assertEqual(torch.get_num_threads(), threads)

    def test_offline_env_and_explicit_path_never_download(self):
        with patch.object(pretrained, "MANIFEST", self.manifest), \
                patch.object(pretrained, "hf_hub_download") as download, \
                patch.object(pretrained, "load_model_from_checkpoint") as load:
            with patch.dict(os.environ, {"CLIQUEFLOWMER_CHECKPOINT": str(self.path)}):
                pretrained.load_pretrained_model()
            with patch.dict(os.environ, {"CLIQUEFLOWMER_CHECKPOINT": "/missing/override.pth"}):
                pretrained.load_pretrained_model(self.path)

        download.assert_not_called()
        self.assertEqual(load.call_count, 2)
        self.assertTrue(all(call.args[0] == self.path for call in load.call_args_list))

    def test_corrupt_download_or_local_override_is_rejected_before_loading(self):
        self.path.write_bytes(b"different checkpoint bytes")
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(pretrained, "MANIFEST", self.manifest), \
                patch.object(pretrained, "hf_hub_download", return_value=str(self.path)), \
                patch.object(pretrained, "load_model_from_checkpoint") as load:
            for path in (None, self.path):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "SHA-256"):
                    pretrained.load_pretrained_model(path)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
