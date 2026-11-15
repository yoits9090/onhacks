#!/usr/bin/env python3
"""Frozen, resumable street-view experiment. Never rents a GPU.

Run --dry-run locally. Actual data downloads/encoding require an explicit
stage. Selection reads tuning labels only; evaluation consumes its locked
selection. See EVALUATION.md and experiment_config.json.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value, immutable=False):
    path = Path(path)
    if path.exists() and immutable:
        if read_json(path) != value:
            raise RuntimeError(f"Immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def check_deadline(config):
    deadline = datetime.fromisoformat(config["deadline_utc"])
    if datetime.now(timezone.utc) >= deadline:
        raise RuntimeError("Experiment deadline reached; preserve/export existing artifacts.")


def source_identity(root, config_path):
    paths = ["scripts/run_experiment.py", "chipoint/baseline.py",
             "chipoint/towers.py", "chipoint/nn.py", "chipoint/load.py",
             "chipoint/geo.py"]
    return {"config_sha256": digest(config_path),
            "implementation": {name: digest(root / name) for name in paths}}


def verified_dataset_file(root, config, name):
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().repo_info(config["dataset_repo"], repo_type="dataset",
                            revision=config["dataset_revision"], files_metadata=True)
    files = {item.rfilename: item for item in info.siblings}
    entry = files[name]
    expected = entry.lfs.sha256 if entry.lfs else None
    if not expected or not entry.size:
        raise RuntimeError(f"Dataset file has no authoritative size/hash: {name}")
    dest = root / "data" / name
    if not dest.exists():
        dest = Path(hf_hub_download(config["dataset_repo"], name, repo_type="dataset",
                                   revision=config["dataset_revision"],
                                   local_dir=root / "data"))
    if dest.stat().st_size != entry.size or digest(dest) != expected:
        raise RuntimeError(f"Dataset file failed verification: {dest}")
    return dest


def make_manifest(metadata_path, archive_path, config):
    """One deterministic image per sequence; disjoint sequences per split."""
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for name in archive.namelist():
            if Path(name).suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            image_id = Path(name).stem
            if image_id in members:
                raise RuntimeError(f"Duplicate archive image ID: {image_id}")
            members[image_id] = name
    best = {}
    seen = set()
    with open(metadata_path, newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "latitude", "longitude", "sequence"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("OSV metadata schema mismatch.")
        for row in reader:
            image_id = row["id"]
            if image_id not in members:
                continue
            if image_id in seen:
                raise RuntimeError(f"Duplicate metadata ID: {image_id}")
            seen.add(image_id)
            sequence = row["sequence"].strip()
            if not sequence or sequence.lower() in ("nan", "none", "null"):
                raise RuntimeError(f"Missing sequence for {image_id}")
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not np.isfinite([lat, lon]).all() or abs(lat) > 90 or abs(lon) > 180:
                raise RuntimeError(f"Invalid coordinates for {image_id}")
            query = {"id": image_id, "sequence": sequence, "lat": lat, "lon": lon,
                     "member": members[image_id], "original_split": "test"}
            key = hash_text(config["image_salt"] + image_id)
            if sequence not in best or key < best[sequence][0]:
                best[sequence] = (key, query)
    if set(members) != seen:
        raise RuntimeError("Archive images missing from the pinned metadata.")
    splits = {"tuning": [], "evaluation": []}
    for sequence, (_, query) in best.items():
        split = ("tuning" if int(hash_text(config["split_salt"] + sequence), 16) % 2 == 0
                 else "evaluation")
        query["split"] = split
        splits[split].append(query)
    count = next((n for n in config["sequence_counts_per_split"]
                  if all(len(rows) >= n for rows in splits.values())), None)
    if count is None:
        raise RuntimeError("Insufficient disjoint sequences for the smallest frozen sample tier.")
    queries = []
    for split, rows in splits.items():
        rows.sort(key=lambda row: (hash_text(config["sample_salt"] + row["sequence"]), row["id"]))
        queries.extend(rows[:count])
    return {"schema": 1, "dataset_repo": config["dataset_repo"],
            "dataset_revision": config["dataset_revision"],
            "archive_file": config["archive_file"],
            "archive_sha256": digest(archive_path),
            "metadata_sha256": digest(metadata_path),
            "queries_per_split": count, "queries": queries}


def prepare(root, run, config, config_path):
    target = run / "manifest.json"
    identity = source_identity(root, config_path)
    if target.exists():
        if read_json(run / "source.json") != identity:
            raise RuntimeError("Run code/config changed; refuse resume.")
        print("Reusing existing immutable manifest.", flush=True)
        return
    check_deadline(config)
    metadata = verified_dataset_file(root, config, config["metadata_file"])
    archive = verified_dataset_file(root, config, config["archive_file"])
    manifest = make_manifest(metadata, archive, config)
    save_json(run / "source.json", identity, immutable=True)
    save_json(target, manifest, immutable=True)
    print(f"Manifest frozen: {manifest['queries_per_split']} sequences per side.", flush=True)


def state_digest(module):
    """Hash the loaded numeric state, including dtype, rather than its name."""
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        # uint8 view also supports bf16, which numpy cannot represent directly.
        h.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def read_batch(path, expected_ids, k, n_rows):
    with np.load(path, allow_pickle=False) as cache:
        ids, sims = cache["gallery_ids"].copy(), cache["similarities"].copy()
        query_ids = cache["query_ids"].tolist()
    if (query_ids != expected_ids or ids.shape != (len(expected_ids), k) or
            sims.shape != ids.shape or ids.dtype != np.int64 or
            sims.dtype != np.float32 or not np.isfinite(sims).all() or
            np.any(ids < 0) or np.any(ids >= n_rows) or
            np.any(np.diff(sims, axis=1) > 0) or
            any(len(np.unique(row)) != k for row in ids)):
        raise RuntimeError(f"Invalid retrieval cache: {path}")
    return ids, sims


def cache(root, run, config):
    import torch
    from PIL import Image
    from chipoint.load import load_proj_head
    from chipoint.towers import Tower
    if not torch.cuda.is_available():
        raise RuntimeError("The cache stage requires CUDA; no silent CPU fallback.")
    manifest = read_json(run / "manifest.json")
    queries = manifest["queries"]
    coords_path = root / "street_gallery/gallery_all_C_so400m256.npy"
    coords = np.load(coords_path, mmap_mode="r")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise RuntimeError("Expected (N,2) shared gallery coordinates.")
    archive_path = root / "data" / config["archive_file"]
    if digest(archive_path) != manifest["archive_sha256"]:
        raise RuntimeError("Archive differs from frozen manifest.")
    k, batch_size = config["retrieval_k"], config["batch_size"]
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "transformers", "open_clip_torch", "numpy"]}
    check_deadline(config)
    for tag in config["towers"]:
        check_deadline(config)
        emb_path = root / "street_gallery" / f"proj_head_gallery_512_{tag}.npy"
        emb = np.load(emb_path, mmap_mode="r")
        if emb.shape != (len(coords), 512) or emb.dtype != np.float16 or len(emb) < k:
            raise RuntimeError(f"Gallery shape/dtype mismatch: {tag}")
        tower = Tower(tag, device="cuda", root=root)
        head, dim = load_proj_head(root / "weights" / f"proj_head_{tag}.pt")
        if dim != tower.dim:
            raise RuntimeError(f"Projection head mismatch: {tag}")
        head = head.to(device="cuda", dtype=torch.float16).eval()
        identity = {"manifest_sha256": digest(run / "manifest.json"),
                    "source_sha256": digest(run / "source.json"),
                    "gallery_sha256": digest(emb_path), "coords_sha256": digest(coords_path),
                    "encoder_numeric_sha256": state_digest(tower.model),
                    "head_numeric_sha256": state_digest(head), "versions": versions,
                    "device": torch.cuda.get_device_name(0),
                    "encoder_dtype": str(tower.compute_dtype), "gallery_dtype": "float16"}
        save_json(run / "cache" / tag / "identity.json", identity, immutable=True)
        # One 5GB gallery resident at a time avoids re-transferring terabytes
        # over PCIe as each query batch scans the reference gallery.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gallery = torch.from_numpy(emb).to("cuda")
        started = time.monotonic()
        raw = projected = similarities = values = indices = None
        with zipfile.ZipFile(archive_path) as archive, torch.inference_mode():
            for start in range(0, len(queries), batch_size):
                check_deadline(config)
                batch = queries[start:start + batch_size]
                expected_ids = [row["id"] for row in batch]
                path = run / "cache" / tag / f"{start:06d}.npz"
                if path.exists():
                    read_batch(path, expected_ids, k, len(coords))
                    continue
                images = []
                for row in batch:
                    with Image.open(io.BytesIO(archive.read(row["member"]))) as image:
                        images.append(image.convert("RGB"))
                raw = tower.encode(images)
                projected = head(raw.to(dtype=torch.float16))
                if not torch.isfinite(projected).all():
                    raise RuntimeError(f"Nonfinite projected embeddings: {tag}")
                similarities = projected.half() @ gallery.T
                values, indices = similarities.topk(k, dim=1)
                save_arrays(path, query_ids=np.asarray(expected_ids),
                            gallery_ids=indices.cpu().numpy().astype(np.int64),
                            similarities=values.float().cpu().numpy())
                read_batch(path, expected_ids, k, len(coords))
                print(f"{tag}: {min(start+batch_size,len(queries))}/{len(queries)} "
                      f"queries cached; elapsed {time.monotonic()-started:.1f}s", flush=True)
        del tower, head, gallery, similarities, values, indices, raw, projected
        torch.cuda.empty_cache()


def fuse(root, run, config):
    from chipoint.baseline import fuse_pools
    manifest = read_json(run / "manifest.json")
    queries = manifest["queries"]
    coords = np.load(root / "street_gallery/gallery_all_C_so400m256.npy", mmap_mode="r")
    all_pools, identities, chunks = {}, {}, {}
    for tag in config["towers"]:
        identities[tag] = read_json(run / "cache" / tag / "identity.json")
        if (identities[tag]["manifest_sha256"] != digest(run / "manifest.json") or
                identities[tag]["source_sha256"] != digest(run / "source.json")):
            raise RuntimeError("Tower cache identity differs from this run.")
        batches = []
        for start in range(0, len(queries), config["batch_size"]):
            path = run / "cache" / tag / f"{start:06d}.npz"
            expected = [row["id"] for row in queries[start:start + config["batch_size"]]]
            batches.append(read_batch(path, expected, config["retrieval_k"], len(coords)))
            chunks[f"{tag}/{path.name}"] = digest(path)
        all_pools[tag] = (np.concatenate([b[0] for b in batches]),
                          np.concatenate([b[1] for b in batches]))
    if any(identity["coords_sha256"] != digest(root / "street_gallery/gallery_all_C_so400m256.npy")
           for identity in identities.values()):
        raise RuntimeError("Shared coordinates differ from the cache.")
    provenance = {"tower_identities": identities, "retrieval_chunks_sha256": chunks,
                  "manifest_sha256": digest(run / "manifest.json")}
    provenance_path, target = run / "fused_identity.json", run / "fused.npz"
    if target.exists():
        saved = read_json(provenance_path)
        if saved["inputs"] != provenance or saved["fused_sha256"] != digest(target):
            raise RuntimeError("Fused cache or its inputs changed.")
        return
    fused_ids, fused_scores = [], []
    for i in range(len(queries)):
        ids, scores = fuse_pools({tag: (p[0][i], p[1][i]) for tag, p in all_pools.items()})
        if len(ids) != 64:
            raise RuntimeError("Expected 64 fused candidates.")
        fused_ids.append(ids)
        fused_scores.append(scores)
    ids = np.stack(fused_ids)
    save_arrays(target, query_ids=np.asarray([row["id"] for row in queries]),
                gallery_ids=ids, coords=np.asarray(coords[ids]),
                similarities=np.stack(fused_scores))
    save_json(provenance_path, {"inputs": provenance, "fused_sha256": digest(target)},
              immutable=True)


def load_fused(run):
    manifest = read_json(run / "manifest.json")
    identity = read_json(run / "fused_identity.json")
    if (identity["fused_sha256"] != digest(run / "fused.npz") or
            identity["inputs"]["manifest_sha256"] != digest(run / "manifest.json")):
        raise RuntimeError("Fused cache integrity failure.")
    with np.load(run / "fused.npz", allow_pickle=False) as data:
        if data["query_ids"].tolist() != [row["id"] for row in manifest["queries"]]:
            raise RuntimeError("Fused query order differs from manifest.")
        coords, sims = data["coords"].copy(), data["similarities"].copy()
    if coords.shape != (len(manifest["queries"]), 64, 2) or sims.shape != coords.shape[:2]:
        raise RuntimeError("Fused cache shape mismatch.")
    return manifest, coords, sims


def predictions(coords, sims, top_n):
    from chipoint.baseline import consensus
    return np.array([consensus(c, s, top_n=top_n) for c, s in zip(coords, sims)])


def select(run, config):
    from chipoint.geo import metrics
    manifest, coords, sims = load_fused(run)
    locked_path = run / "selected.json"
    if locked_path.exists():
        locked = read_json(locked_path)
        if locked["fused_sha256"] != digest(run / "fused.npz"):
            raise RuntimeError("Selection belongs to a different cache.")
        print("Reusing locked tuning selection.", flush=True)
        return
    mask = np.array([row["split"] == "tuning" for row in manifest["queries"]])
    truth = np.array([[row["lat"], row["lon"]] for row in manifest["queries"]])[mask]
    results = []
    for depth in config["candidate_top_n"]:
        pred = predictions(coords[mask], sims[mask], depth)
        results.append({"top_n": depth, **metrics(pred[:, 0], pred[:, 1],
                                                  truth[:, 0], truth[:, 1])})
    winner = min(results, key=lambda row: (-row["geoscore"],
                 row["top_n"] != config["baseline_top_n"], row["top_n"]))
    save_json(locked_path, {"fused_sha256": digest(run / "fused.npz"),
                           "top_n": winner["top_n"], "tuning_metrics": results},
              immutable=True)
    print(f"Tuning winner locked: top_n={winner['top_n']}.", flush=True)


def paired_interval(deltas, config):
    deltas = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(config["bootstrap_seed"])
    means = []
    for start in range(0, config["bootstrap_replicates"], 250):
        count = min(250, config["bootstrap_replicates"] - start)
        indices = rng.integers(0, len(deltas), (count, len(deltas)))
        means.extend(deltas[indices].mean(axis=1))
    return np.quantile(means, [0.025, 0.975]).tolist()


def evaluate(run, config):
    from chipoint.geo import GEOSCALE, haversine_km, metrics
    manifest, coords, sims = load_fused(run)
    locked = read_json(run / "selected.json")
    if locked["fused_sha256"] != digest(run / "fused.npz"):
        raise RuntimeError("Selection/cache mismatch.")
    if (run / "report.json").exists():
        report = read_json(run / "report.json")
        if report["selection_sha256"] != digest(run / "selected.json"):
            raise RuntimeError("Report/selection mismatch.")
        print("Reusing frozen report; no re-scoring.", flush=True)
        return
    rows = [row for row in manifest["queries"] if row["split"] == "evaluation"]
    if len({row["sequence"] for row in manifest["queries"]}) != len(manifest["queries"]):
        raise RuntimeError("Sequences are duplicated across queries/splits.")
    mask = np.array([row["split"] == "evaluation" for row in manifest["queries"]])
    truth = np.array([[row["lat"], row["lon"]] for row in rows])
    base = predictions(coords[mask], sims[mask], config["baseline_top_n"])
    candidate = predictions(coords[mask], sims[mask], locked["top_n"])
    base_metrics = metrics(base[:, 0], base[:, 1], truth[:, 0], truth[:, 1])
    candidate_metrics = metrics(candidate[:, 0], candidate[:, 1], truth[:, 0], truth[:, 1])
    base_error = haversine_km(base[:, 0], base[:, 1], truth[:, 0], truth[:, 1])
    candidate_error = haversine_km(candidate[:, 0], candidate[:, 1], truth[:, 0], truth[:, 1])
    deltas = 5000 * (np.exp(-candidate_error / GEOSCALE) - np.exp(-base_error / GEOSCALE))
    interval = paired_interval(deltas, config)
    accepted = bool(deltas.mean() > 0 and interval[0] > 0 and
                    candidate_metrics["acc@25km"] >= base_metrics["acc@25km"])
    report = {"scope": "OSV-5M archive-00 development subset; not official benchmark",
              "selection_sha256": digest(run / "selected.json"),
              "evaluation_queries": len(rows), "evaluation_sequences": len(rows),
              "baseline": base_metrics, "selected": candidate_metrics,
              "selected_top_n": locked["top_n"],
              "mean_geoscore_delta": float(deltas.mean()),
              "paired_sequence_bootstrap_95pct": interval,
              "accept_candidate": accepted, "failed_images": 0,
              "decision": "use selected candidate" if accepted else "retain baseline"}
    save_json(run / "prediction_config.json",
              {"top_n": locked["top_n"] if accepted else config["baseline_top_n"],
               "towers": config["towers"], "retrieval_k": config["retrieval_k"],
               "model_revision": config["model_revision"],
               "kind": "consensus calibration, not encoder fine-tuning"}, immutable=True)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["id", "sequence", "baseline_lat", "baseline_lon",
                     "selected_lat", "selected_lon", "baseline_km", "selected_km"])
    for row, b, c, be, ce in zip(rows, base, candidate, base_error, candidate_error):
        writer.writerow([row["id"], row["sequence"], *b, *c, be, ce])
    pred_path = run / "predictions.csv"
    tmp = pred_path.with_suffix(".csv.part")
    tmp.write_text(stream.getvalue())
    os.replace(tmp, pred_path)
    report["predictions_sha256"] = digest(pred_path)
    report["prediction_config_sha256"] = digest(run / "prediction_config.json")
    save_json(run / "report.json", report, immutable=True)
    print(json.dumps(report, indent=2), flush=True)


def export_examples(root, run):
    """Present three preselected evaluation queries, never the best errors."""
    report = read_json(run / "report.json")
    if (report["selection_sha256"] != digest(run / "selected.json") or
            report["predictions_sha256"] != digest(run / "predictions.csv") or
            report["prediction_config_sha256"] != digest(run / "prediction_config.json")):
        raise RuntimeError("Presentation inputs differ from the frozen report.")
    manifest = read_json(run / "manifest.json")
    identity = read_json(run / "fused_identity.json")
    if identity["inputs"]["manifest_sha256"] != digest(run / "manifest.json"):
        raise RuntimeError("Presentation manifest differs from the evaluated manifest.")
    out = run / "demo_examples"
    target = out / "examples.json"
    if target.exists():
        for row in read_json(target):
            if digest(out / row["image"]) != row["image_sha256"]:
                raise RuntimeError("Exported presentation image changed.")
        print("Reusing frozen presentation examples.", flush=True)
        return
    from PIL import Image
    rows = [row for row in manifest["queries"] if row["split"] == "evaluation"][:3]
    if len(rows) < 3:
        raise RuntimeError("Need three evaluation queries for the presentation export.")
    with open(run / "predictions.csv", newline="") as stream:
        paired = {row["id"]: row for row in csv.DictReader(stream)}
    archive_path = root / "data" / manifest["archive_file"]
    if digest(archive_path) != manifest["archive_sha256"]:
        raise RuntimeError("Presentation image archive differs from the frozen manifest.")
    out.mkdir(parents=True, exist_ok=True)
    examples = []
    with zipfile.ZipFile(archive_path) as archive:
        for i, query in enumerate(rows):
            prediction = paired[query["id"]]
            if prediction["sequence"] != query["sequence"]:
                raise RuntimeError("Presentation ID/sequence alignment error.")
            filename = f"example-{i}.jpg"
            with Image.open(io.BytesIO(archive.read(query["member"]))) as image:
                encoded = io.BytesIO()
                image.convert("RGB").save(encoded, format="JPEG", quality=90)
            image_path = out / filename
            payload = encoded.getvalue()
            if image_path.exists() and image_path.read_bytes() != payload:
                raise RuntimeError(f"Refusing to replace different image: {image_path}")
            tmp = image_path.with_suffix(".jpg.part")
            tmp.write_bytes(payload)
            os.replace(tmp, image_path)
            fields = {key: float(prediction[key]) for key in
                      ("baseline_lat", "baseline_lon", "selected_lat", "selected_lon",
                       "baseline_km", "selected_km")}
            deployed = "candidate" if report["accept_candidate"] else "baseline"
            prefix = "selected" if report["accept_candidate"] else "baseline"
            examples.append({"id": query["id"], "sequence": query["sequence"],
                             "image": filename, "image_sha256": digest(image_path),
                             "true_lat": query["lat"], "true_lon": query["lon"],
                             **fields, "prediction_lat": fields[f"{prefix}_lat"],
                             "prediction_lon": fields[f"{prefix}_lon"],
                             "deployed": deployed})
    save_json(target, examples, immutable=True)
    print("Exported first three evaluation examples, not selected by performance.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "cache", "select", "evaluate", "export", "all"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "experiment_config.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/experiment-v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    stages = ["prepare", "cache", "select", "evaluate", "export"] if args.stage == "all" else [args.stage]
    if args.dry_run:
        print(json.dumps({"stages": stages, "config": config, "run_dir": str(args.run_dir),
                          "rents_gpu": False}, indent=2))
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage not in ("prepare", "all"):
        if read_json(args.run_dir / "source.json") != source_identity(args.root, args.config):
            raise RuntimeError("Run source/config changed; refuse resume.")
    for stage in stages:
        print(f"== {stage} ==", flush=True)
        if stage == "prepare":
            prepare(args.root, args.run_dir, config, args.config)
        elif stage == "cache":
            cache(args.root, args.run_dir, config)
            fuse(args.root, args.run_dir, config)
        elif stage == "select":
            select(args.run_dir, config)
        elif stage == "evaluate":
            evaluate(args.run_dir, config)
        else:
            export_examples(args.root, args.run_dir)


if __name__ == "__main__":
    main()
