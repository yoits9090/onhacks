"""Synthetic controls for the preregistered second attempt, not accuracy."""
import csv
import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiment as exp
from chipoint.baseline import fuse_pools
from chipoint.calibration import weighted_fuse_pools

spec = importlib.util.spec_from_file_location("fusion", ROOT / "scripts/calibrate_fusion.py")
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)


class WeightTests(unittest.TestCase):
    def test_selection_guard_and_baseline_ties(self):
        cfg = exp.read_json(ROOT / "fusion_config.json")
        baseline = {"top_n": 10, "tower_weights": dict.fromkeys(cfg["towers"], 1),
                    "acc@25km": 40, "geoscore": 3900}
        bad = {**baseline, "top_n": 5, "acc@25km": 39, "geoscore": 5000}
        good = {**baseline, "top_n": 20, "geoscore": 3910}
        self.assertEqual(fusion.choose_candidate([bad, good, baseline], cfg), good)
        self.assertEqual(fusion.choose_candidate([{**good, "geoscore": 3900}, baseline], cfg), baseline)

    def test_all_one_exact_reference_and_controlled_weight_change(self):
        rng = np.random.default_rng(551)
        for _ in range(20):
            pools = {tag: (rng.choice(400, 200, replace=False),
                           np.sort(rng.random(200))[::-1].astype(np.float32))
                     for tag in ("so400m256", "gopt", "dinov3")}
            ref = fuse_pools(pools)
            got = weighted_fuse_pools(pools, dict.fromkeys(pools, 1.0))
            np.testing.assert_array_equal(ref[0], got[0])
            np.testing.assert_array_equal(ref[1], got[1])
        pools = {"a": (np.array([0, 1]), np.array([1., 0.])),
                 "b": (np.array([1, 0]), np.array([1., 0.])),
                 "c": (np.array([1, 0]), np.array([1., 0.]))}
        self.assertEqual(weighted_fuse_pools(pools, {"a": 1, "b": 1, "c": 1}, keep=2)[0][0], 1)
        self.assertEqual(weighted_fuse_pools(pools, {"a": 4, "b": 1, "c": 1}, keep=2)[0][0], 0)

    def test_bad_weight_contract_fails(self):
        pools = {"a": (np.array([0, 1]), np.array([1., 0.]))}
        for weights in ({}, {"a": 0}, {"a": -1}, {"a": float("nan")}, {"a": 1, "b": 2}):
            with self.assertRaises(ValueError):
                weighted_fuse_pools(pools, weights)


class FreshCohortTests(unittest.TestCase):
    def test_excludes_both_old_splits_and_respects_global_split(self):
        cfg = exp.read_json(ROOT / "fusion_config.json")
        cfg["evaluation_sequence_counts"] = [10, 5]
        rows = [{"id": f"{i}-{j}", "sequence": f"sequence-{i}", "latitude": 0, "longitude": 0}
                for i in range(100) for j in range(2)]
        original = {"queries": [{"sequence": f"sequence-{i}",
                                  "split": "tuning" if i % 2 else "evaluation"} for i in range(20)]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            def make(rows):
                with open(path / "test.csv", "w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                with zipfile.ZipFile(path / "01.zip", "w") as archive:
                    for row in rows:
                        archive.writestr(row["id"] + ".jpg", b"fixture")
                return fusion.fresh_manifest(path / "test.csv", path / "01.zip", original, cfg)
            a, b = make(rows), make(list(reversed(rows)))
            self.assertEqual(a["queries"], b["queries"])
            self.assertEqual(len(a["queries"]), 10)
            seqs = [q["sequence"] for q in a["queries"]]
            self.assertEqual(len(set(seqs)), 10)
            self.assertFalse(set(seqs) & {q["sequence"] for q in original["queries"]})
            self.assertTrue(all(int(exp.hash_text(cfg["split_salt"] + s), 16) % 2 == 1 for s in seqs))
            cfg["evaluation_sequence_counts"] = [1000]
            with self.assertRaisesRegex(RuntimeError, "Insufficient fresh"):
                make(rows)


class EvaluationTests(unittest.TestCase):
    def test_fresh_paired_report_and_idempotent_resume(self):
        cfg = exp.read_json(ROOT / "fusion_config.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, base = root / "run", root / "base"
            seqs = [f"fresh-{i}" for i in range(100)
                    if int(exp.hash_text(cfg["split_salt"] + f"fresh-{i}"), 16) % 2 == 1][:10]
            manifest = {"queries": [{"id": str(i), "sequence": seq, "lat": 0, "lon": 0,
                                      "split": "evaluation"} for i, seq in enumerate(seqs)]}
            exp.save_json(run / "source.json", {"fixture": True})
            exp.save_json(run / "tuning_lock.json", {
                "source_sha256": exp.digest(run / "source.json"),
                "winner": {"top_n": 5, "tower_weights": {"so400m256": 1, "gopt": 0.5, "dinov3": 2}},
                "baseline": {"top_n": 10, "tower_weights": dict.fromkeys(cfg["towers"], 1.0)}})
            exp.save_json(base / "manifest.json", {"queries": [{"sequence": "old-sequence"}]})
            coords = np.zeros((10, 64, 2)); coords[:, 5:, 1] = 1
            scores = np.ones((10, 64), dtype=np.float32)
            exp.save_arrays(run / "fused.npz", coords=coords, similarities=scores)
            exp.save_json(run / "fused_identity.json", {"fixture": True})
            (root / "street_gallery").mkdir()
            np.save(root / "street_gallery/gallery_all_C_so400m256.npy", np.zeros((256, 2)))
            with mock.patch.object(exp, "load_fused", return_value=(manifest, coords, scores)), \
                 mock.patch.object(fusion, "pools_for", return_value={}), \
                 mock.patch.object(fusion, "fuse_queries", return_value=(coords, scores)):
                fusion.evaluate(root, base, run, cfg)
                report = exp.read_json(run / "report.json")
                self.assertTrue(report["accept_candidate"])
                self.assertGreater(report["paired_sequence_bootstrap_95pct"][0], 0)
                self.assertEqual(report["evaluation_queries"], 10)
                self.assertEqual(exp.read_json(run / "prediction_config.json")["top_n"], 5)
                before = (run / "report.json").read_bytes()
                with mock.patch.object(exp, "predictions", side_effect=AssertionError("no re-scoring")):
                    fusion.evaluate(root, base, run, cfg)
                self.assertEqual((run / "report.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
