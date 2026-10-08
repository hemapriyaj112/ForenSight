"""
v2_1/artifact_audit.py
======================
Repository / auxiliary-artifact audit for the V2.1 experiment.

Produces, for every persisted Day-2..Day-15 artifact that V2.1 was asked to
look for:

  1. artifact path
  2. number of rows
  3. available features (column names, summarised by family)
  4. label availability
  5. sample-ID availability
  6. overlap with every other audited artifact (by sample id and, where
     available, by content hash)
  7. population tier: development / locked / fresh diagnostic /
     honest holdout / closed

Read-only. No production code is touched.
"""
from __future__ import annotations

from . import features as feat
from . import populations as pops

#: Human-readable population tier per artifact key.
TIER = {
    "p4b_frame": "development (frozen V2 rank-reference population + 3 unscored)",
    "p4b_video": "development (frozen V2 rank-reference population + 3 unscored)",
    "p4b_audio": "development (frozen V2 rank-reference population + 3 unscored)",
    "p4b_split_manifest": "manifest (all splits; LOCKED rows are closed)",
    "p4b_dataset_manifest": "manifest (all splits; LOCKED rows are closed)",
    "p4b_fusion_inputs": "development (frozen V2 rank-reference population)",
    "p4b_fusion_predictions": "development (frozen V2 rank-reference population)",
    "d7_manifest": "manifest (all 29: DEV 16 / OTHER 8 / HOLDOUT 5)",
    "d7_frame": "mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)",
    "d7_window": "mixed (DEV usable, HOLDOUT 5 closed)",
    "d7_x_agg": "mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)",
    "d7_a_agg": "mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)",
    "d7_audio_robustness": "mixed (DEV 16 usable, HOLDOUT 5 closed)",
    "d7_identity": "mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)",
    "locked_raw": "LOCKED - CLOSED",
    "locked_inputs": "LOCKED - CLOSED",
    "d7_fresh_predictions": "fresh diagnostic (Day-7) - descriptive only",
    "d7_baseline_predictions": "fresh diagnostic (Day-7) - descriptive only",
    "d8_failure_mode_table": "fresh diagnostic (Day-8) - descriptive only",
    "d9_dev_predictions": "development (Day-9 DEV 16)",
    "d9_holdout_predictions": "HONEST HOLDOUT - CLOSED",
    "d10_dev_predictions": "development (Day-10 DEV 16)",
    "d10_holdout_predictions": "HONEST HOLDOUT - CLOSED",
    "d11_descriptors": "mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)",
    "d11_dev_predictions": "development (Day-11 DEV 16)",
    "d11_holdout_predictions": "HONEST HOLDOUT - CLOSED",
    "d5_crossvalidation": "development OOF (historical, frozen reference)",
    "candidate_freeze": "frozen V2 candidate definition",
    "final_summary": "historical V2 final summary",
}

FEATURE_FAMILY = {
    "x_": "Xception visual score",
    "a_": "AASIST audio score",
    "rank": "frozen-reference rank",
    "fused": "frozen fused score",
    "frac": "temporal high-score fraction",
    "longest_run": "temporal persistence",
    "n_transitions": "temporal persistence",
    "burst": "temporal persistence",
    "half_": "temporal drift",
    "face": "face/evidence availability",
    "id_": "Day-10 visual identity",
    "area_": "Day-10 visual identity",
    "centroid_": "Day-10 visual identity",
    "audio_": "Day-9 audio robustness",
    "domain_": "Day-9 domain risk",
    "bitrate": "Day-9 audio robustness",
    "resolution": "Day-9 audio robustness",
    "n_window": "AASIST window count",
    "video_score": "Xception video aggregate",
    "audio_fake": "AASIST video aggregate",
    "audio_bonafide": "AASIST video aggregate",
    "fake_probability": "Xception per-frame score",
    "face_confidence": "Xception per-frame face evidence",
    "faces_in_frame": "Xception per-frame face evidence",
    "fake_direction_score": "AASIST per-window score",
    "confidence": "classifier confidence",
}


def summarise_features(columns: list[str]) -> dict:
    families: dict[str, list[str]] = {}
    for col in columns:
        key = next((k for k in FEATURE_FAMILY if col.startswith(k)), "other")
        families.setdefault(key, []).append(col)
    return {k: {"n": len(v), "columns": v} for k, v in sorted(families.items())}


def audit(root=None) -> dict:
    """Full artifact audit. Returns a JSON-serialisable dict."""
    ids_by_key: dict[str, set] = {}
    hashes_by_key: dict[str, set] = {}
    rows_by_key: dict[str, list[dict]] = {}
    rows_by_artifact: dict[str, dict] = {}
    entries = []

    for key, rel in feat.ARTIFACTS.items():
        path = feat.artifact_path(key, root)
        record = {
            "artifact_key": key,
            "path": rel,
            "exists": path.exists(),
            "population_tier": TIER.get(key, "unclassified"),
        }
        rows_by_artifact[key] = record
        if path.suffix == ".json":
            import json
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                record["n_rows"] = None
                record["top_level_keys"] = sorted(payload)[:20] if isinstance(
                    payload, dict) else None
                record["has_sample_ids"] = "oof_sample_ids" in path.read_text(
                    encoding="utf-8")
            entries.append(record)
            continue
        if not path.exists():
            record["n_rows"] = 0
            entries.append(record)
            continue
        rows = feat.read_csv_rows(path)
        rows_by_key[key] = rows
        cols = list(rows[0].keys()) if rows else []
        record["n_rows"] = len(rows)
        record["n_columns"] = len(cols)
        record["columns"] = cols
        record["feature_families"] = summarise_features(cols)
        record["label_column"] = next(
            (c for c in ("true_label", "label", "verdict", "error") if c in cols),
            None)
        record["has_labels"] = record["label_column"] is not None
        if record["has_labels"]:
            counts: dict[str, int] = {}
            for r in rows:
                counts[str(r.get(record["label_column"]))] = (
                    counts.get(str(r.get(record["label_column"])), 0) + 1)
            record["label_counts"] = counts
        record["has_sample_ids"] = "sample_id" in cols
        record["has_content_hash"] = "md5" in cols
        if record["has_sample_ids"]:
            sids = {str(r["sample_id"]) for r in rows if r.get("sample_id")}
            ids_by_key[key] = sids
            record["n_unique_sample_ids"] = len(sids)
        if record["has_content_hash"]:
            hashes_by_key[key] = {
                str(r["md5"]) for r in rows if r.get("md5")}
        entries.append(record)

    # --- overlap matrix on sample ids ------------------------------------- #
    overlap = []
    keys = sorted(ids_by_key)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = ids_by_key[a] & ids_by_key[b]
            if shared:
                overlap.append({
                    "artifact_a": a, "artifact_b": b,
                    "n_shared_sample_ids": len(shared),
                    "shared_sample_ids": sorted(shared),
                    "byte_identical_md5": bool(
                        hashes_by_key.get(a) and hashes_by_key.get(b)
                        and (hashes_by_key[a] & hashes_by_key[b])),
                })

    # --- cross-pool byte-identity check ------------------------------------ #
    p4b_md5 = hashes_by_key.get("p4b_dataset_manifest", set())
    d7_md5 = hashes_by_key.get("d7_manifest", set())
    cross_pool = {
        "p4b_md5_count": len(p4b_md5),
        "d7_md5_count": len(d7_md5),
        "shared_md5": sorted(p4b_md5 & d7_md5),
        "byte_identical_overlap": bool(p4b_md5 & d7_md5),
    }

    closed_report = []
    for pop in pops.CLOSED_POPULATIONS:
        ids = set()
        for sid in pop["ids"]:
            ids.add(sid.split()[0])
        present_in = sorted(k for k, s in ids_by_key.items() if ids & s)
        closed_report.append({**pop, "present_in_artifacts": present_in})

    return {
        "artifacts": entries,
        "sample_id_overlap": overlap,
        "cross_pool_content_hash": cross_pool,
        "closed_populations": closed_report,
        "development_pools": pops.pool_definitions(),
        "sample_ids_by_artifact": {k: sorted(v) for k, v in ids_by_key.items()},
        "evidence_row_counts": {
            row_key: rows_by_artifact.get(key, {}).get("n_rows", 0)
            for key, row_key in (("p4b_frame", "P4B_FRAMES"),
                                 ("d7_frame", "D7_FRAMES"),
                                 ("d7_window", "D7_AASIST_WINDOWS"))},
    }
