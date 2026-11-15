#!/usr/bin/env python3
"""Second preregistered attempt: old tuning data, new disjoint evaluation."""
import argparse
import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_experiment as exp
from chipoint.baseline import consensus
from chipoint.calibration import weighted_fuse_pools
from chipoint.geo import GEOSCALE, haversine_km, metrics


def source(root, config_path, base):
    identity = exp.source_identity(root, config_path)
    for name in ("scripts/calibrate_fusion.py", "chipoint/calibration.py"):
        identity["implementation"][name] = exp.digest(root / name)
    identity["base_manifest_sha256"] = exp.digest(base / "manifest.json")
    identity["base_cache_identity_sha256"] = exp.digest(base / "fused_identity.json")
    return identity


def pools_for(run, manifest, config):
    pools = {}
    for tag in config["towers"]:
        batches = []
        for start in range(0, len(manifest["queries"]), config["batch_size"]):
            path = run / "cache" / tag / f"{start:06d}.npz"
            expected = [q["id"] for q in manifest["queries"][start:start + config["batch_size"]]]
            # The source .npy row count is not shipped to the laptop; actual
            # bounds were checked during encoding. Rechecked against coords below.
            batches.append(exp.read_batch(path, expected, config["retrieval_k"], 2**63 - 1))
        pools[tag] = tuple(np.concatenate([batch[i] for batch in batches]) for i in (0, 1))
    return pools


def fuse_queries(pools, gallery_coords, weights, query_indices):
    coords, scores = [], []
    for i in query_indices:
        ids, values = weighted_fuse_pools(
            {tag: (pool[0][i], pool[1][i]) for tag, pool in pools.items()}, weights)
        if len(ids) != 64 or np.any(ids >= len(gallery_coords)):
            raise RuntimeError("Invalid weighted-fusion candidate IDs.")
        coords.append(np.asarray(gallery_coords[ids]))
        scores.append(values)
    return np.stack(coords), np.stack(scores)


def is_baseline(candidate, config):
    return (candidate["top_n"] == config["baseline_top_n"] and
            all(value == 1 for value in candidate["tower_weights"].values()))


def choose_candidate(candidates, config):
    baseline = next(c for c in candidates if is_baseline(c, config))
    eligible = [c for c in candidates if c["acc@25km"] >= baseline["acc@25km"]]
    return min(eligible, key=lambda c: (-c["geoscore"], not is_baseline(c, config)))


def tune(root, base, run, config, config_path):
    expected_source = source(root, config_path, base)
    exp.save_json(run / "source.json", expected_source, immutable=True)
    exp.save_json(run / "spec.json", config, immutable=True)
    lock = run / "tuning_lock.json"
    if lock.exists():
        return exp.read_json(lock)
    original_source = exp.read_json(base / "source.json")
    if original_source != exp.source_identity(root, root / "experiment_config.json"):
        raise RuntimeError("Original experiment code/config changed; refuse reuse.")
    original_identity = exp.read_json(base / "fused_identity.json")
    for name, expected in original_identity["inputs"]["retrieval_chunks_sha256"].items():
        if exp.digest(base / "cache" / name) != expected:
            raise RuntimeError(f"Original cache changed: {name}")
    manifest, _, _ = exp.load_fused(base)
    indices = [i for i, q in enumerate(manifest["queries"]) if q["split"] == "tuning"]
    truth = np.array([[manifest["queries"][i]["lat"], manifest["queries"][i]["lon"]]
                      for i in indices])
    pools = pools_for(base, manifest, config)
    coords_path = root / "street_gallery/gallery_all_C_so400m256.npy"
    if any(identity["coords_sha256"] != exp.digest(coords_path)
           for identity in original_identity["inputs"]["tower_identities"].values()):
        raise RuntimeError("Coordinate file differs from the original retrieval cache.")
    gallery_coords = np.load(coords_path, mmap_mode="r")
    candidates = []
    for gopt_weight in config["relative_tower_weights"]:
        for dino_weight in config["relative_tower_weights"]:
            exp.check_deadline(config)
            weights = {"so400m256": 1.0, "gopt": gopt_weight, "dinov3": dino_weight}
            coords, scores = fuse_queries(pools, gallery_coords, weights, indices)
            for depth in config["candidate_top_n"]:
                predicted = exp.predictions(coords, scores, depth)
                candidates.append({
                    "top_n": depth, "tower_weights": weights,
                    **metrics(predicted[:, 0], predicted[:, 1], truth[:, 0], truth[:, 1])})
            print(f"Tuning weights {weights} complete ({len(candidates)}/36 configs).", flush=True)
    baseline = next(c for c in candidates if is_baseline(c, config))
    prior_baseline = next(c for c in exp.read_json(base / "selected.json")["tuning_metrics"]
                          if c["top_n"] == config["baseline_top_n"])
    if (abs(baseline["geoscore"] - prior_baseline["geoscore"]) > 1e-6 or
            abs(baseline["acc@25km"] - prior_baseline["acc@25km"]) > 1e-9):
        raise RuntimeError("Equal-weight control does not reproduce the frozen tuning baseline.")
    winner = choose_candidate(candidates, config)
    result = {"source_sha256": exp.digest(run / "source.json"),
              "tuning_sequences": len(indices), "winner": winner, "baseline": baseline,
              "candidates": candidates}
    exp.save_json(lock, result, immutable=True)
    print("Winner locked before new evaluation cohort is prepared: "
          + json.dumps(winner), flush=True)
    return result


def fresh_manifest(metadata, archive, original_manifest, config):
    excluded = {q["sequence"] for q in original_manifest["queries"]}
    with zipfile.ZipFile(archive) as z:
        names = {}
        for name in z.namelist():
            if Path(name).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                key = Path(name).stem
                if key in names:
                    raise RuntimeError("Duplicate image ID in fresh archive.")
                names[key] = name
    seen, best = set(), {}
    with open(metadata, newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "sequence", "latitude", "longitude"}.issubset(reader.fieldnames or []):
            raise RuntimeError("Fresh metadata schema mismatch.")
        for row in reader:
            image_id = row["id"]
            if image_id not in names:
                continue
            if image_id in seen:
                raise RuntimeError("Duplicate fresh metadata ID.")
            seen.add(image_id)
            seq = row["sequence"].strip()
            if not seq or seq.lower() in ("nan", "none", "null"):
                raise RuntimeError("Missing fresh sequence.")
            if seq in excluded or int(exp.hash_text(config["split_salt"] + seq), 16) % 2 != 1:
                continue
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not np.isfinite([lat, lon]).all() or abs(lat) > 90 or abs(lon) > 180:
                raise RuntimeError("Invalid fresh coordinates.")
            query = {"id": image_id, "sequence": seq, "lat": lat, "lon": lon,
                     "member": names[image_id], "split": "evaluation", "original_split": "test"}
            key = exp.hash_text(config["image_salt"] + image_id)
            if seq not in best or key < best[seq][0]:
                best[seq] = (key, query)
    if seen != set(names):
        raise RuntimeError("Fresh archive is not aligned with metadata.")
    rows = [item[1] for item in best.values()]
    rows.sort(key=lambda q: (exp.hash_text(config["sample_salt"] + q["sequence"]), q["id"]))
    n = next((n for n in config["evaluation_sequence_counts"] if len(rows) >= n), None)
    if n is None:
        raise RuntimeError("Insufficient fresh evaluation sequences; do not resize post hoc.")
    return {"schema": 1, "dataset_repo": config["dataset_repo"],
            "dataset_revision": config["dataset_revision"],
            "archive_file": config["archive_file"], "archive_sha256": exp.digest(archive),
            "metadata_sha256": exp.digest(metadata), "queries_per_split": n,
            "excluded_original_sequences": len(excluded), "queries": rows[:n]}


def prepare(root, base, run, config):
    if (run / "manifest.json").exists():
        return
    exp.check_deadline(config)
    metadata = exp.verified_dataset_file(root, config, config["metadata_file"])
    archive = exp.verified_dataset_file(root, config, config["archive_file"])
    manifest = fresh_manifest(metadata, archive, exp.read_json(base / "manifest.json"), config)
    exp.save_json(run / "manifest.json", manifest, immutable=True)
    print(f"Fresh evaluation manifest frozen: {len(manifest['queries'])} sequences.", flush=True)


def evaluate(root, base, run, config):
    lock = exp.read_json(run / "tuning_lock.json")
    if exp.digest(run / "source.json") != lock["source_sha256"]:
        raise RuntimeError("Tuning lock/source mismatch.")
    if (run / "report.json").exists():
        print("Reusing immutable v2 report.", flush=True)
        return
    winner = lock["winner"]
    manifest, base_coords, base_scores = exp.load_fused(run)
    old_sequences = {q["sequence"] for q in exp.read_json(base / "manifest.json")["queries"]}
    queries = manifest["queries"]
    if (len({q["sequence"] for q in queries}) != len(queries) or
            any(q["sequence"] in old_sequences or q["split"] != "evaluation" or
                int(exp.hash_text(config["split_salt"] + q["sequence"]), 16) % 2 != 1
                for q in queries)):
        raise RuntimeError("Fresh evaluation overlaps an original sequence or tuning split.")
    capture, identity_path = run / "candidate_fused.npz", run / "candidate_identity.json"
    inputs = {"tuning_lock_sha256": exp.digest(run / "tuning_lock.json"),
              "fresh_cache_identity_sha256": exp.digest(run / "fused_identity.json")}
    if capture.exists():
        identity = exp.read_json(identity_path)
        if identity["inputs"] != inputs or identity["sha256"] != exp.digest(capture):
            raise RuntimeError("Captured weighted candidates or their inputs changed.")
        with np.load(capture, allow_pickle=False) as data:
            candidate_coords, candidate_scores = data["coords"].copy(), data["similarities"].copy()
    else:
        pools = pools_for(run, manifest, config)
        gallery_coords = np.load(root / "street_gallery/gallery_all_C_so400m256.npy", mmap_mode="r")
        candidate_coords, candidate_scores = fuse_queries(
            pools, gallery_coords, winner["tower_weights"], range(len(queries)))
        exp.save_arrays(capture, coords=candidate_coords, similarities=candidate_scores)
        exp.save_json(identity_path, {"inputs": inputs, "sha256": exp.digest(capture)}, immutable=True)
    baseline = exp.predictions(base_coords, base_scores, config["baseline_top_n"])
    candidate = exp.predictions(candidate_coords, candidate_scores, winner["top_n"])
    truth = np.array([[q["lat"], q["lon"]] for q in queries])
    base_metrics = metrics(baseline[:, 0], baseline[:, 1], truth[:, 0], truth[:, 1])
    candidate_metrics = metrics(candidate[:, 0], candidate[:, 1], truth[:, 0], truth[:, 1])
    base_error = haversine_km(baseline[:, 0], baseline[:, 1], truth[:, 0], truth[:, 1])
    candidate_error = haversine_km(candidate[:, 0], candidate[:, 1], truth[:, 0], truth[:, 1])
    delta = 5000 * (np.exp(-candidate_error / GEOSCALE) - np.exp(-base_error / GEOSCALE))
    interval = exp.paired_interval(delta, config)
    accepted = bool(delta.mean() > 0 and interval[0] > 0 and
                    candidate_metrics["acc@25km"] >= base_metrics["acc@25km"])
    exp.save_json(run / "selected.json",
                  {"top_n": winner["top_n"], "tower_weights": winner["tower_weights"],
                   "fused_sha256": exp.digest(run / "fused.npz"),
                   "tuning_lock_sha256": exp.digest(run / "tuning_lock.json")}, immutable=True)
    deployed = winner if accepted else lock["baseline"]
    exp.save_json(run / "prediction_config.json",
                  {"top_n": deployed["top_n"], "tower_weights": deployed["tower_weights"],
                   "model_revision": config["model_revision"], "towers": config["towers"],
                   "retrieval_k": config["retrieval_k"],
                   "kind": "tower-fusion consensus calibration, not encoder training"}, immutable=True)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["id", "sequence", "baseline_lat", "baseline_lon",
                     "selected_lat", "selected_lon", "baseline_km", "selected_km"])
    for q, b, c, be, ce in zip(queries, baseline, candidate, base_error, candidate_error):
        writer.writerow([q["id"], q["sequence"], *b, *c, be, ce])
    tmp = run / "predictions.csv.part"
    tmp.write_text(stream.getvalue())
    os.replace(tmp, run / "predictions.csv")
    report = {"scope": "Fresh OSV-5M archive-01 subset; all original 4000 sequences excluded; not official benchmark",
              "selection_sha256": exp.digest(run / "selected.json"),
              "tuning_lock_sha256": exp.digest(run / "tuning_lock.json"),
              "candidate_fused_sha256": exp.digest(capture),
              "evaluation_queries": len(queries), "evaluation_sequences": len(queries),
              "baseline": base_metrics, "selected": candidate_metrics,
              "selected_top_n": winner["top_n"], "selected_tower_weights": winner["tower_weights"],
              "mean_geoscore_delta": float(delta.mean()),
              "paired_sequence_bootstrap_95pct": interval, "accept_candidate": accepted,
              "failed_images": 0, "decision": "use selected candidate" if accepted else "retain baseline",
              "predictions_sha256": exp.digest(run / "predictions.csv"),
              "prediction_config_sha256": exp.digest(run / "prediction_config.json")}
    exp.save_json(run / "report.json", report, immutable=True)
    print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["tune", "prepare", "cache", "evaluate", "export", "all"])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--config", type=Path, default=ROOT / "fusion_config.json")
    p.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/experiment-v2")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    config = exp.read_json(a.config)
    if a.dry_run:
        print(json.dumps(config, indent=2))
        return
    base = a.root / config["base_run"]
    a.run_dir.mkdir(parents=True, exist_ok=True)
    stages = ["tune", "prepare", "cache", "evaluate", "export"] if a.stage == "all" else [a.stage]
    for stage in stages:
        print(f"== fusion-v2 {stage} ==", flush=True)
        if stage == "tune":
            lock = tune(a.root, base, a.run_dir, config, a.config)
        else:
            if exp.read_json(a.run_dir / "source.json") != source(a.root, a.config, base):
                raise RuntimeError("V2 source/config changed; refuse resume.")
            lock = exp.read_json(a.run_dir / "tuning_lock.json")
        if is_baseline(lock["winner"], config):
            exp.save_json(a.run_dir / "tuning_only_result.json",
                          {"decision": "retain baseline; no eligible tuning gain",
                           "tuning_lock_sha256": exp.digest(a.run_dir / "tuning_lock.json")},
                          immutable=True)
            print("Baseline won tuning; stop without claiming new evaluation.", flush=True)
            return
        if stage == "prepare":
            prepare(a.root, base, a.run_dir, config)
        elif stage == "cache":
            exp.cache(a.root, a.run_dir, config)
            for tag in config["towers"]:
                old = exp.read_json(base / "cache" / tag / "identity.json")
                new = exp.read_json(a.run_dir / "cache" / tag / "identity.json")
                for key in ("encoder_numeric_sha256", "head_numeric_sha256",
                            "gallery_sha256", "coords_sha256", "versions", "encoder_dtype"):
                    if old[key] != new[key]:
                        raise RuntimeError(f"Frozen tower state changed: {tag}/{key}")
            exp.fuse(a.root, a.run_dir, config)
        elif stage == "evaluate":
            evaluate(a.root, base, a.run_dir, config)
        elif stage == "export":
            exp.export_examples(a.root, a.run_dir)


if __name__ == "__main__":
    main()
