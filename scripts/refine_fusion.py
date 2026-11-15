#!/usr/bin/env python3
"""Fixed local refinement against saved v2, on a third disjoint cohort."""
import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import calibrate_fusion as fc
import run_experiment as exp
from chipoint.geo import GEOSCALE, haversine_km, metrics


def same_parameters(a, b):
    return a["top_n"] == b["top_n"] and a["tower_weights"] == b["tower_weights"]


def choose(candidates, reference):
    eligible = [c for c in candidates if c["acc@25km"] >= reference["acc@25km"]]
    return min(eligible, key=lambda c: (-c["geoscore"], not same_parameters(c, reference)))


def source(root, config_path, cfg):
    result = fc.source(root, config_path, root / cfg["base_run"])
    result["implementation"]["scripts/refine_fusion.py"] = exp.digest(
        root / "scripts/refine_fusion.py")
    result["reference_manifest_sha256"] = exp.digest(
        root / cfg["reference_run"] / "manifest.json")
    result["reference_tuning_lock_sha256"] = exp.digest(
        root / cfg["reference_run"] / "tuning_lock.json")
    result["reference_checkpoint_sha256"] = exp.digest(
        root / cfg["reference_checkpoint"] / "prediction_config.json")
    return result


def tune(root, run, cfg, config_path):
    base = root / cfg["base_run"]
    ref_run = root / cfg["reference_run"]
    checkpoint = exp.read_json(root / cfg["reference_checkpoint"] / "prediction_config.json")
    prior = exp.read_json(ref_run / "tuning_lock.json")
    if not exp.read_json(ref_run / "report.json")["accept_candidate"]:
        raise RuntimeError("Reference must be the accepted, saved v2 checkpoint.")
    if (not same_parameters(checkpoint, prior["winner"]) or
            checkpoint != exp.read_json(ref_run / "prediction_config.json")):
        raise RuntimeError("Saved checkpoint differs from accepted reference.")
    exp.save_json(run / "source.json", source(root, config_path, cfg), immutable=True)
    exp.save_json(run / "spec.json", cfg, immutable=True)
    if (run / "tuning_lock.json").exists():
        return exp.read_json(run / "tuning_lock.json")
    if exp.read_json(base / "source.json") != exp.source_identity(root, root / "experiment_config.json"):
        raise RuntimeError("Original encoder/retrieval implementation changed.")
    frozen = exp.read_json(base / "fused_identity.json")
    for name, h in frozen["inputs"]["retrieval_chunks_sha256"].items():
        if exp.digest(base / "cache" / name) != h:
            raise RuntimeError(f"Frozen tuning cache changed: {name}")
    manifest, _, _ = exp.load_fused(base)
    indices = [i for i, q in enumerate(manifest["queries"]) if q["split"] == "tuning"]
    truth = np.array([[manifest["queries"][i]["lat"], manifest["queries"][i]["lon"]]
                      for i in indices])
    pools = fc.pools_for(base, manifest, cfg)
    coords_path = root / "street_gallery/gallery_all_C_so400m256.npy"
    coords_hash = exp.digest(coords_path)
    if any(v["coords_sha256"] != coords_hash
           for v in frozen["inputs"]["tower_identities"].values()):
        raise RuntimeError("Frozen coordinate file changed.")
    gallery = np.load(coords_path, mmap_mode="r")
    candidates = []
    for gw in cfg["gopt_weights"]:
        for dw in cfg["dinov3_weights"]:
            exp.check_deadline(cfg)
            weights = {"so400m256": 1.0, "gopt": gw, "dinov3": dw}
            cc, ss = fc.fuse_queries(pools, gallery, weights, indices)
            for depth in cfg["candidate_top_n"]:
                parameters = {"top_n": depth, "tower_weights": weights}
                cached = next((c for c in prior["candidates"]
                               if same_parameters(c, parameters)), None)
                if cached is not None:
                    candidates.append(cached.copy())
                else:
                    p = exp.predictions(cc, ss, depth)
                    candidates.append({**parameters,
                                       **metrics(p[:, 0], p[:, 1], truth[:, 0], truth[:, 1])})
            print(f"Refinement tuning weights {weights} complete.", flush=True)
    reference = next(c for c in candidates if same_parameters(c, checkpoint))
    if reference != prior["winner"]:
        raise RuntimeError("Cached reference tuning metrics changed.")
    winner = choose(candidates, reference)
    lock = {"source_sha256": exp.digest(run / "source.json"),
            "tuning_sequences": len(indices), "reference": reference,
            "winner": winner, "candidates": candidates}
    exp.save_json(run / "tuning_lock.json", lock, immutable=True)
    print("Refinement locked before fresh cohort: " + json.dumps(winner), flush=True)
    return lock


def combined_previous(root, cfg):
    a = exp.read_json(root / cfg["base_run"] / "manifest.json")
    b = exp.read_json(root / cfg["reference_run"] / "manifest.json")
    queries = a["queries"] + b["queries"]
    if len({q["sequence"] for q in queries}) != len(queries):
        raise RuntimeError("Previous cohorts overlap unexpectedly.")
    return {"queries": queries}


def prepare(root, run, cfg):
    if (run / "manifest.json").exists():
        return
    exp.check_deadline(cfg)
    metadata = exp.verified_dataset_file(root, cfg, cfg["metadata_file"])
    archive = exp.verified_dataset_file(root, cfg, cfg["archive_file"])
    manifest = fc.fresh_manifest(metadata, archive, combined_previous(root, cfg), cfg)
    exp.save_json(run / "manifest.json", manifest, immutable=True)
    print(f"Fresh v3 cohort frozen: {len(manifest['queries'])} sequences.", flush=True)


def intervals(deltas, cfg):
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    means = []
    for start in range(0, cfg["bootstrap_replicates"], 250):
        n = min(250, cfg["bootstrap_replicates"] - start)
        means.extend(deltas[rng.integers(0, len(deltas), (n, len(deltas)))].mean(axis=1))
    alpha = cfg["familywise_alpha"] / cfg["planned_attempts"]
    return (np.quantile(means, [.025, .975]).tolist(),
            np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist())


def accept(delta, adjusted_interval, candidate_metrics, baseline_metrics):
    return bool(delta > 0 and adjusted_interval[0] > 0 and
                candidate_metrics["acc@25km"] >= baseline_metrics["acc@25km"])


def capture(run, name, pools, gallery, weights, count, inputs):
    path, identity_path = run / f"{name}.npz", run / f"{name}_identity.json"
    expected = {**inputs, "tower_weights": weights}
    if path.exists():
        identity = exp.read_json(identity_path)
        if identity["inputs"] != expected or identity["sha256"] != exp.digest(path):
            raise RuntimeError(f"Changed captured inputs: {name}")
        with np.load(path, allow_pickle=False) as data:
            return data["coords"].copy(), data["similarities"].copy()
    coords, scores = fc.fuse_queries(pools, gallery, weights, range(count))
    exp.save_arrays(path, coords=coords, similarities=scores)
    exp.save_json(identity_path, {"inputs": expected, "sha256": exp.digest(path)}, immutable=True)
    return coords, scores


def evaluate(root, run, cfg):
    lock = exp.read_json(run / "tuning_lock.json")
    if lock["source_sha256"] != exp.digest(run / "source.json"):
        raise RuntimeError("Tuning lock/source mismatch.")
    if (run / "report.json").exists():
        report = exp.read_json(run / "report.json")
        for name, key in (("predictions.csv", "predictions_sha256"),
                          ("prediction_config.json", "prediction_config_sha256"),
                          ("selected.json", "selection_sha256")):
            if exp.digest(run / name) != report[key]:
                raise RuntimeError(f"Completed report artifact changed: {name}")
        return
    manifest, _, _ = exp.load_fused(run)
    queries = manifest["queries"]
    excluded = {q["sequence"] for q in combined_previous(root, cfg)["queries"]}
    if (len({q["sequence"] for q in queries}) != len(queries) or
            any(q["sequence"] in excluded or q["split"] != "evaluation" or
                int(exp.hash_text(cfg["split_salt"] + q["sequence"]), 16) % 2 != 1
                for q in queries)):
        raise RuntimeError("Fresh cohort overlaps previous sequences or tuning partition.")
    pools = fc.pools_for(run, manifest, cfg)
    gallery = np.load(root / "street_gallery/gallery_all_C_so400m256.npy", mmap_mode="r")
    inputs = {"tuning_lock_sha256": exp.digest(run / "tuning_lock.json"),
              "fresh_cache_identity_sha256": exp.digest(run / "fused_identity.json")}
    reference, winner = lock["reference"], lock["winner"]
    bc, bs = capture(run, "baseline_fused", pools, gallery,
                     reference["tower_weights"], len(queries), inputs)
    cc, cs = capture(run, "candidate_fused", pools, gallery,
                     winner["tower_weights"], len(queries), inputs)
    baseline = exp.predictions(bc, bs, reference["top_n"])
    candidate = exp.predictions(cc, cs, winner["top_n"])
    truth = np.array([[q["lat"], q["lon"]] for q in queries])
    bm = metrics(baseline[:, 0], baseline[:, 1], truth[:, 0], truth[:, 1])
    cm = metrics(candidate[:, 0], candidate[:, 1], truth[:, 0], truth[:, 1])
    be = haversine_km(baseline[:, 0], baseline[:, 1], truth[:, 0], truth[:, 1])
    ce = haversine_km(candidate[:, 0], candidate[:, 1], truth[:, 0], truth[:, 1])
    delta = 5000 * (np.exp(-ce / GEOSCALE) - np.exp(-be / GEOSCALE))
    ordinary, adjusted = intervals(delta, cfg)
    accepted = accept(delta.mean(), adjusted, cm, bm)
    exp.save_json(run / "selected.json",
                  {"top_n": winner["top_n"], "tower_weights": winner["tower_weights"],
                   "fused_sha256": exp.digest(run / "fused.npz"),
                   "tuning_lock_sha256": exp.digest(run / "tuning_lock.json")}, immutable=True)
    checkpoint = exp.read_json(root / cfg["reference_checkpoint"] / "prediction_config.json")
    deployed = ({**checkpoint, "top_n": winner["top_n"],
                 "tower_weights": winner["tower_weights"],
                 "kind": "refined tower-fusion calibration, not encoder training"}
                if accepted else checkpoint)
    exp.save_json(run / "prediction_config.json", deployed, immutable=True)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["id", "sequence", "baseline_lat", "baseline_lon",
                     "selected_lat", "selected_lon", "baseline_km", "selected_km"])
    for q, b, c, bd, cd in zip(queries, baseline, candidate, be, ce):
        writer.writerow([q["id"], q["sequence"], *b, *c, bd, cd])
    tmp = run / "predictions.csv.part"
    tmp.write_text(stream.getvalue())
    os.replace(tmp, run / "predictions.csv")
    report = {
        "scope": "Fresh OSV-5M archive-02 subset; all 6000 previous sequences excluded; reference is accepted v2, not original baseline",
        "baseline_kind": "accepted fusion-v2 checkpoint", "baseline_top_n": reference["top_n"],
        "baseline_tower_weights": reference["tower_weights"],
        "baseline_prediction_config": checkpoint,
        "evaluation_queries": len(queries), "evaluation_sequences": len(queries),
        "baseline": bm, "selected": cm, "selected_top_n": winner["top_n"],
        "selected_tower_weights": winner["tower_weights"],
        "mean_geoscore_delta": float(delta.mean()),
        "paired_sequence_bootstrap_95pct": ordinary,
        "paired_sequence_bootstrap_adjusted": adjusted,
        "familywise_confidence_percent": 100 * (1 - cfg["familywise_alpha"] / cfg["planned_attempts"]),
        "planned_attempts": cfg["planned_attempts"], "accept_candidate": accepted,
        "decision": "use refined candidate" if accepted else "retain accepted v2 checkpoint",
        "failed_images": 0, "selection_sha256": exp.digest(run / "selected.json"),
        "tuning_lock_sha256": exp.digest(run / "tuning_lock.json"),
        "baseline_fused_sha256": exp.digest(run / "baseline_fused.npz"),
        "candidate_fused_sha256": exp.digest(run / "candidate_fused.npz"),
        "predictions_sha256": exp.digest(run / "predictions.csv"),
        "prediction_config_sha256": exp.digest(run / "prediction_config.json")}
    exp.save_json(run / "report.json", report, immutable=True)
    print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tune", "prepare", "cache", "evaluate", "export", "all"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "refinement_config.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/experiment-v3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = exp.read_json(args.config)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stages = ["tune", "prepare", "cache", "evaluate", "export"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"== refinement-v3 {stage} ==", flush=True)
        if stage == "tune":
            lock = tune(args.root, args.run_dir, cfg, args.config)
        else:
            if exp.read_json(args.run_dir / "source.json") != source(args.root, args.config, cfg):
                raise RuntimeError("Refinement source/config/reference changed.")
            lock = exp.read_json(args.run_dir / "tuning_lock.json")
        if same_parameters(lock["winner"], lock["reference"]):
            exp.save_json(args.run_dir / "tuning_only_result.json",
                          {"decision": "retain accepted v2; no eligible tuning gain",
                           "tuning_lock_sha256": exp.digest(args.run_dir / "tuning_lock.json")},
                          immutable=True)
            print("Saved v2 won tuning; no new final comparison needed.", flush=True)
            return
        if stage == "prepare":
            prepare(args.root, args.run_dir, cfg)
        elif stage == "cache":
            exp.cache(args.root, args.run_dir, cfg)
            for tag in cfg["towers"]:
                old = exp.read_json(args.root / cfg["base_run"] / "cache" / tag / "identity.json")
                new = exp.read_json(args.run_dir / "cache" / tag / "identity.json")
                for key in ("encoder_numeric_sha256", "head_numeric_sha256", "gallery_sha256",
                            "coords_sha256", "versions", "encoder_dtype"):
                    if old[key] != new[key]:
                        raise RuntimeError(f"Frozen model state changed: {tag}/{key}")
            exp.fuse(args.root, args.run_dir, cfg)
        elif stage == "evaluate":
            evaluate(args.root, args.run_dir, cfg)
        elif stage == "export":
            exp.export_examples(args.root, args.run_dir)


if __name__ == "__main__":
    main()
