"""Lead-authored checks for experiment separation, caching and arithmetic.

Synthetic fixtures only. These checks do not measure a model's accuracy.
"""
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("experiment", ROOT / "scripts/run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def config():
    value = experiment.read_json(ROOT / "experiment_config.json")
    value["sequence_counts_per_split"] = [4, 2]
    return value


def score_fixture(path, equal=False, evaluation_longitude=0):
    queries = [{"id": str(i), "sequence": f"seq-{i}",
                "split": "tuning" if i < 4 else "evaluation",
                "lat": 0, "lon": 0 if i < 4 else evaluation_longitude}
               for i in range(8)]
    experiment.save_json(path / "manifest.json", {"queries": queries})
    coords = np.zeros((8, 64, 2), dtype=np.float64)
    if not equal:
        coords[:, 5:10, 1] = 1
        coords[:, 10:, 1] = 1.2
    experiment.save_arrays(path / "fused.npz", coords=coords,
                           query_ids=np.asarray([str(i) for i in range(8)]),
                           similarities=np.ones((8, 64), dtype=np.float32))
    experiment.save_json(path / "fused_identity.json",
                         {"fused_sha256": experiment.digest(path / "fused.npz"),
                          "inputs": {"manifest_sha256": experiment.digest(path / "manifest.json")}})


class ManifestTests(unittest.TestCase):
    def make_inputs(self, path, reverse=False, missing_sequence=False):
        rows = []
        for i in range(64):
            for j in range(2):
                rows.append({"id": f"{i}-{j}", "sequence": "" if missing_sequence else f"seq{i}",
                             "latitude": 10 + i / 100, "longitude": 20})
        if reverse:
            rows.reverse()
        metadata, archive = path / "test.csv", path / "test.zip"
        with open(metadata, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with zipfile.ZipFile(archive, "w") as stream:
            for row in rows:
                stream.writestr(f"test/{row['id']}.jpg", b"not decoded by manifest builder")
        return metadata, archive

    def test_one_image_per_sequence_order_independent_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = experiment.make_manifest(*self.make_inputs(root), config())
            second = experiment.make_manifest(*self.make_inputs(root, reverse=True), config())
            self.assertEqual(first["queries"], second["queries"])
            self.assertEqual(first["queries_per_split"], 4)
            tuning = {q["sequence"] for q in first["queries"] if q["split"] == "tuning"}
            evaluation = {q["sequence"] for q in first["queries"] if q["split"] == "evaluation"}
            self.assertEqual(len(tuning), 4)
            self.assertEqual(len(evaluation), 4)
            self.assertFalse(tuning & evaluation)
            self.assertTrue(all(q["original_split"] == "test" for q in first["queries"]))

    def test_missing_sequence_fails_instead_of_silent_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "Missing sequence"):
                experiment.make_manifest(*self.make_inputs(Path(tmp), missing_sequence=True), config())

    def test_insufficient_sequences_does_not_resize_after_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config()
            cfg["sequence_counts_per_split"] = [10000]
            with self.assertRaisesRegex(RuntimeError, "Insufficient"):
                experiment.make_manifest(*self.make_inputs(Path(tmp)), cfg)


class SelectionTests(unittest.TestCase):
    def test_presentation_uses_first_three_frozen_queries_and_resume(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            score_fixture(run)
            (root / "data").mkdir()
            archive = root / "data/test.zip"
            manifest = experiment.read_json(run / "manifest.json")
            with zipfile.ZipFile(archive, "w") as stream:
                for row in manifest["queries"]:
                    row["member"] = f"test/{row['id']}.jpg"
                    payload = io.BytesIO()
                    Image.new("RGB", (32, 24), (int(row["id"]) * 20, 40, 60)).save(payload, format="JPEG")
                    stream.writestr(row["member"], payload.getvalue())
            manifest["archive_file"] = "test.zip"
            manifest["archive_sha256"] = experiment.digest(archive)
            experiment.save_json(run / "manifest.json", manifest)
            identity = experiment.read_json(run / "fused_identity.json")
            identity["inputs"]["manifest_sha256"] = experiment.digest(run / "manifest.json")
            experiment.save_json(run / "fused_identity.json", identity)
            experiment.select(run, config())
            experiment.evaluate(run, config())
            experiment.export_examples(root, run)
            exported = experiment.read_json(run / "demo_examples/examples.json")
            self.assertEqual([row["id"] for row in exported], ["4", "5", "6"])
            self.assertTrue(all(row["deployed"] == "candidate" for row in exported))
            self.assertTrue(all(row["prediction_lon"] == row["selected_lon"] for row in exported))
            with mock.patch("PIL.Image.open", side_effect=AssertionError("must not re-render")):
                experiment.export_examples(root, run)
            (run / "predictions.csv").write_text("tampered")
            with self.assertRaisesRegex(RuntimeError, "Presentation inputs differ"):
                experiment.export_examples(root, run)

    def test_evaluation_labels_cannot_change_tuning_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            score_fixture(a, evaluation_longitude=0)
            score_fixture(b, evaluation_longitude=100)
            experiment.select(a, config())
            experiment.select(b, config())
            left, right = experiment.read_json(a / "selected.json"), experiment.read_json(b / "selected.json")
            self.assertEqual(left["top_n"], 5)
            self.assertEqual(left["top_n"], right["top_n"])
            self.assertEqual(left["tuning_metrics"], right["tuning_metrics"])

    def test_known_positive_difference_and_frozen_report(self):
        from chipoint.geo import EARTH_R_KM, GEOSCALE
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            score_fixture(run)
            experiment.select(run, config())
            experiment.evaluate(run, config())
            report = experiment.read_json(run / "report.json")
            # Baseline has five points at 0 deg, five at 1 deg: spherical
            # mean is exactly longitude 0.5 deg; the selected top-five is 0.
            distance = EARTH_R_KM * np.deg2rad(0.5)
            expected_delta = 5000 * (1 - np.exp(-distance / GEOSCALE))
            self.assertAlmostEqual(report["mean_geoscore_delta"], expected_delta, places=9)
            np.testing.assert_allclose(report["paired_sequence_bootstrap_95pct"],
                                       [expected_delta, expected_delta], atol=1e-9)
            self.assertTrue(report["accept_candidate"])
            self.assertEqual(report["evaluation_queries"], 4)
            before = (run / "report.json").read_bytes()
            with mock.patch.object(experiment, "predictions", side_effect=AssertionError("must not rescore")):
                experiment.evaluate(run, config())
                experiment.select(run, config())
            self.assertEqual((run / "report.json").read_bytes(), before)

    def test_no_gain_keeps_baseline_and_does_not_claim_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            score_fixture(run, equal=True)
            experiment.select(run, config())
            experiment.evaluate(run, config())
            report = experiment.read_json(run / "report.json")
            self.assertEqual(report["selected_top_n"], 10)
            self.assertFalse(report["accept_candidate"])
            self.assertEqual(report["mean_geoscore_delta"], 0)
            self.assertEqual(report["paired_sequence_bootstrap_95pct"], [0, 0])
            self.assertEqual(experiment.read_json(run / "prediction_config.json")["top_n"], 10)

    def test_overlapping_sequences_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            score_fixture(run)
            manifest = experiment.read_json(run / "manifest.json")
            manifest["queries"][-1]["sequence"] = manifest["queries"][0]["sequence"]
            experiment.save_json(run / "manifest.json", manifest)
            identity = experiment.read_json(run / "fused_identity.json")
            identity["inputs"]["manifest_sha256"] = experiment.digest(run / "manifest.json")
            experiment.save_json(run / "fused_identity.json", identity)
            experiment.select(run, config())
            with self.assertRaisesRegex(RuntimeError, "Sequences are duplicated"):
                experiment.evaluate(run, config())


class CacheTests(unittest.TestCase):
    def test_fusion_reuses_frozen_inputs_and_rejects_changed_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = Path(tmp), config()
            run = root / "run"
            score_fixture(run)
            experiment.save_json(run / "source.json", {"fixture": True})
            (root / "street_gallery").mkdir()
            coords = np.column_stack((np.arange(256) / 100, np.zeros(256)))
            coords_path = root / "street_gallery/gallery_all_C_so400m256.npy"
            np.save(coords_path, coords)
            query_ids = np.array([str(i) for i in range(8)])
            cfg["retrieval_k"], cfg["batch_size"] = 200, 16
            for offset, tag in enumerate(cfg["towers"]):
                identity = {"manifest_sha256": experiment.digest(run / "manifest.json"),
                            "source_sha256": experiment.digest(run / "source.json"),
                            "coords_sha256": experiment.digest(coords_path)}
                experiment.save_json(run / "cache" / tag / "identity.json", identity)
                ids = np.tile((np.arange(200) + offset) % 256, (8, 1)).astype(np.int64)
                sims = np.tile(np.linspace(0.9, 0.2, 200, dtype=np.float32), (8, 1))
                experiment.save_arrays(run / "cache" / tag / "000000.npz",
                                       query_ids=query_ids, gallery_ids=ids, similarities=sims)
            # score_fixture supplied a fused cache for selection tests. For
            # this fixture build fresh fusion only from retrieval chunks.
            (run / "fused.npz").unlink()
            (run / "fused_identity.json").unlink()
            experiment.fuse(root, run, cfg)
            before = (run / "fused.npz").read_bytes()
            with mock.patch("chipoint.baseline.fuse_pools",
                            side_effect=AssertionError("must not re-fuse")):
                experiment.fuse(root, run, cfg)
            self.assertEqual(before, (run / "fused.npz").read_bytes())
            path = run / "cache" / cfg["towers"][0] / "000000.npz"
            ids, sims = experiment.read_batch(path, query_ids.tolist(), 200, 256)
            sims[0, 0] += 0.001
            experiment.save_arrays(path, query_ids=query_ids,
                                   gallery_ids=ids, similarities=sims)
            with self.assertRaisesRegex(RuntimeError, "Fused cache or its inputs changed"):
                experiment.fuse(root, run, cfg)

    def test_valid_batch_reuse_and_wrong_query_order_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.npz"
            ids = np.array([[1, 2], [3, 4]], dtype=np.int64)
            sims = np.array([[0.8, 0.5], [0.9, 0.2]], dtype=np.float32)
            experiment.save_arrays(path, query_ids=np.array(["a", "b"]),
                                   gallery_ids=ids, similarities=sims)
            actual_ids, actual_sims = experiment.read_batch(path, ["a", "b"], 2, 5)
            np.testing.assert_array_equal(actual_ids, ids)
            np.testing.assert_array_equal(actual_sims, sims)
            with self.assertRaisesRegex(RuntimeError, "Invalid retrieval cache"):
                experiment.read_batch(path, ["b", "a"], 2, 5)
            path.write_bytes(b"corrupted archive")
            with self.assertRaises((ValueError, OSError)):
                experiment.read_batch(path, ["a", "b"], 2, 5)

    def test_nonfinite_duplicate_and_out_of_range_candidates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.npz"
            for ids, sims in [([[1, 1]], [[0.9, 0.5]]),
                              ([[1, 5]], [[0.9, 0.5]]),
                              ([[1, 2]], [[np.nan, 0.5]]),
                              ([[1, 2]], [[0.5, 0.9]])]:
                experiment.save_arrays(path, query_ids=np.array(["a"]),
                                       gallery_ids=np.asarray(ids, dtype=np.int64),
                                       similarities=np.asarray(sims, dtype=np.float32))
                with self.assertRaisesRegex(RuntimeError, "Invalid retrieval cache"):
                    experiment.read_batch(path, ["a"], 2, 5)

    def test_immutable_artifact_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.json"
            experiment.save_json(path, {"top_n": 10}, immutable=True)
            with self.assertRaisesRegex(RuntimeError, "Immutable artifact differs"):
                experiment.save_json(path, {"top_n": 5}, immutable=True)
            self.assertEqual(experiment.read_json(path), {"top_n": 10})


if __name__ == "__main__":
    unittest.main(verbosity=2)
