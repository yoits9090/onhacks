"""Synthetic controls for v3; no benchmark accuracy claims."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiment as exp

spec = importlib.util.spec_from_file_location("refine", ROOT / "scripts/refine_fusion.py")
refine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refine)


class RefinementTests(unittest.TestCase):
    def test_reference_is_saved_v2_not_equal_weights(self):
        reference = {"top_n": 5, "tower_weights": {"so400m256": 1, "gopt": 2, "dinov3": .5},
                     "geoscore": 4100, "acc@25km": 40}
        bad = {**reference, "top_n": 3, "geoscore": 4500, "acc@25km": 39}
        tie = {**reference, "top_n": 7}
        self.assertEqual(refine.choose([bad, tie, reference], reference), reference)
        good = {**reference, "top_n": 3, "geoscore": 4110}
        self.assertEqual(refine.choose([reference, bad, good], reference), good)

    def test_adjusted_gate_and_intervals(self):
        cfg = exp.read_json(ROOT / "refinement_config.json")
        a, b = refine.intervals(np.ones(10), cfg)
        self.assertEqual(a, [1, 1])
        self.assertEqual(b, [1, 1])
        a, b = refine.intervals(np.linspace(-2, 4, 100), cfg)
        self.assertLessEqual(b[0], a[0])
        self.assertGreaterEqual(b[1], a[1])
        self.assertFalse(refine.accept(1, [-.1, 2], {"acc@25km": 41}, {"acc@25km": 40}))
        self.assertFalse(refine.accept(1, [.1, 2], {"acc@25km": 39}, {"acc@25km": 40}))
        self.assertTrue(refine.accept(1, [.1, 2], {"acc@25km": 40}, {"acc@25km": 40}))

    def test_combined_exclusions_include_both_previous_cohorts(self):
        cfg = exp.read_json(ROOT / "refinement_config.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp.save_json(root / cfg["base_run"] / "manifest.json",
                          {"queries": [{"sequence": "old-a"}]})
            exp.save_json(root / cfg["reference_run"] / "manifest.json",
                          {"queries": [{"sequence": "old-b"}]})
            self.assertEqual([q["sequence"] for q in refine.combined_previous(root, cfg)["queries"]],
                             ["old-a", "old-b"])
            exp.save_json(root / cfg["reference_run"] / "manifest.json",
                          {"queries": [{"sequence": "old-a"}]})
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                refine.combined_previous(root, cfg)

    def test_paired_report_uses_weighted_checkpoint_as_control_and_freezes(self):
        cfg = exp.read_json(ROOT / "refinement_config.json")
        with tempfile.TemporaryDirectory() as tmp:
            root, run = Path(tmp), Path(tmp) / "run"
            reference = {"top_n": 5, "tower_weights": {"so400m256": 1, "gopt": 2, "dinov3": .5}}
            candidate = {"top_n": 3, "tower_weights": {"so400m256": 1, "gopt": 3, "dinov3": .25}}
            checkpoint = {**reference, "kind": "fixture saved v2",
                          "towers": cfg["towers"], "retrieval_k": 200,
                          "model_revision": cfg["model_revision"]}
            exp.save_json(root / cfg["reference_checkpoint"] / "prediction_config.json", checkpoint)
            exp.save_json(root / cfg["base_run"] / "manifest.json",
                          {"queries": [{"sequence": "old-a"}]})
            exp.save_json(root / cfg["reference_run"] / "manifest.json",
                          {"queries": [{"sequence": "old-b"}]})
            seqs = [f"fresh-{i}" for i in range(100)
                    if int(exp.hash_text(cfg["split_salt"] + f"fresh-{i}"), 16) % 2 == 1][:10]
            manifest = {"queries": [{"id": str(i), "sequence": s, "lat": 0, "lon": 0,
                                      "split": "evaluation"} for i, s in enumerate(seqs)]}
            exp.save_json(run / "source.json", {"fixture": True})
            exp.save_json(run / "tuning_lock.json", {
                "source_sha256": exp.digest(run / "source.json"),
                "reference": reference, "winner": candidate})
            zeros, scores = np.zeros((10, 64, 2)), np.ones((10, 64))
            exp.save_arrays(run / "fused.npz", coords=zeros, similarities=scores)
            exp.save_json(run / "fused_identity.json", {"fixture": True})
            (root / "street_gallery").mkdir()
            np.save(root / "street_gallery/gallery_all_C_so400m256.npy", np.zeros((1, 2)))

            def capture(run, name, pools, gallery, weights, count, inputs):
                c = zeros.copy()
                if name == "baseline_fused":
                    self.assertEqual(weights, reference["tower_weights"])
                    c[:, :, 1] = 1  # v2 is 111km wrong; equal-weight dummy above is exact
                else:
                    self.assertEqual(weights, candidate["tower_weights"])
                exp.save_arrays(run / f"{name}.npz", coords=c, similarities=scores)
                return c, scores

            with mock.patch.object(exp, "load_fused", return_value=(manifest, zeros, scores)), \
                 mock.patch.object(refine.fc, "pools_for", return_value={}), \
                 mock.patch.object(refine, "capture", side_effect=capture):
                refine.evaluate(root, run, cfg)
                report = exp.read_json(run / "report.json")
                self.assertTrue(report["accept_candidate"])
                self.assertEqual(report["baseline_top_n"], 5)
                self.assertEqual(report["baseline_prediction_config"], checkpoint)
                self.assertGreater(report["baseline"]["mean_km"], 100)
                self.assertEqual(report["selected"]["mean_km"], 0)
                self.assertGreater(report["paired_sequence_bootstrap_adjusted"][0], 0)
                self.assertEqual(exp.read_json(run / "prediction_config.json")["top_n"], 3)
                before = (run / "report.json").read_bytes()
                with mock.patch.object(exp, "predictions", side_effect=AssertionError("no re-scoring")):
                    refine.evaluate(root, run, cfg)
                self.assertEqual((run / "report.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
