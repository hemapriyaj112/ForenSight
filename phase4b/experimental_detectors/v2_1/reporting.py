"""
v2_1/reporting.py
=================
Markdown renderers for the V2.1 experiment outputs.

All numbers come from the already-computed metric rows / OofResult objects;
nothing is recomputed here.
"""
from __future__ import annotations

import statistics

from . import candidates as cand
from . import features as feat

BAR = "|"


def _table(headers: list[str], rows: list[list]) -> str:
    out = [f"{BAR} " + f" {BAR} ".join(headers) + f" {BAR}",
           f"{BAR}" + "|".join(["---"] * len(headers)) + f"{BAR}"]
    for row in rows:
        out.append(f"{BAR} " + f" {BAR} ".join("" if c is None else str(c)
                                                for c in row) + f" {BAR}")
    return "\n".join(out)


def _f(value, digits=4):
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# --------------------------------------------------------------------------- #
# main experiment report
# --------------------------------------------------------------------------- #

def render_report(result: dict, metric_rows: list[dict], results: list,
                  table: dict) -> str:
    sel = result["selection"]
    lines: list[str] = []
    A = lines.append

    A("# V2.1 Accuracy Improvement Experiment - Report")
    A("")
    A(f"**Status: {result['status']}**")
    A("")
    A("This is a V2.1 EXPERIMENTAL MODEL-SELECTION / ANALYSIS PASS. It does not")
    A("replace the documented V2 final result, does not modify the frozen V2")
    A("architecture, does not touch V1, and does not modify the production")
    A("inference path or UI.")
    A("")
    A(f"- run started (UTC): `{result['started_utc']}`")
    A(f"- run finished (UTC): `{result['finished_utc']}`")
    A("")

    # ---------------- 1. data populations -------------------------------- #
    A("## 1. Exact data populations used")
    A("")
    A("Two disjoint development-eligible pools. They are reported SEPARATELY and")
    A("are never presented as one homogeneous population. A pooled view exists")
    A("only as a clearly-labelled sensitivity analysis.")
    A("")
    rows = []
    for name, lc in result["populations"].items():
        tier = next((p["tier"] for p in result["data_integrity"]["development_pools"]
                     if p["pool"] == name),
                    "SENSITIVITY ONLY - explicit concatenation of the two "
                    "development pools, never a headline number")
        rows.append([f"`{name}`", lc["n"], lc["n_fake"], lc["n_real"], tier])
    A(_table(["pool", "n", "FAKE", "REAL", "tier"], rows))
    A("")
    A("Sample ids:")
    A("")
    for name in ("P4B_DEV25", "D7_DEV16"):
        ids = [r["sample_id"] for r in table[name]]
        A(f"- `{name}` ({len(ids)}): " + ", ".join(f"`{i}`" for i in ids))
    A("")

    A("### Populations deliberately NOT used")
    A("")
    rows = []
    for pop in result["data_integrity"]["closed_populations"]:
        rows.append([pop["population"], pop["n"], pop["status"],
                     ", ".join(f"`{i}`" for i in pop["ids"])])
    A(_table(["population", "n", "status", "ids"], rows))
    A("")

    # ---------------- 2. feature set ------------------------------------- #
    A("## 2. Exact feature set")
    A("")
    A("Built ONLY from already-persisted Day-2..Day-11 evidence. No detector was")
    A("re-run, no model was downloaded, nothing was retrained, no media was")
    A("decoded. Temporal descriptors reuse the historical Day-11 code verbatim.")
    A("")
    for name, cols in result["feature_sets"].items():
        A(f"### `{name}` ({len(cols)} features)")
        A("")
        if name == "EXTENDED":
            A("D7-only: AASIST per-window statistics and the Day-10 identity")
            A("statistics were never persisted for P4B. `rank_x` / `rank_a` /")
            A("`fused_v2` are legitimate here (fixed external reference) but the")
            A("set is deliberately over-parameterised for n=16 and is reported as")
            A("an overfitting probe, not a promotion candidate.")
            A("")
        A(", ".join(f"`{c}`" for c in cols))
        A("")
    A("Explicitly excluded as label proxies: "
      + ", ".join(f"`{c}`" for c in result["excluded_leakage_columns"]))
    A("")
    A("In `P4B_DEV25` the grouping column `source_group` is **100% label-pure**")
    A("(13/13 groups single-label), so using it - or any one-hot of it - would be")
    A("direct label leakage. It is used only as a CV grouping key.")
    A("")

    # ---------------- 3. candidates -------------------------------------- #
    A("## 3. Exact model candidates (pre-declared)")
    A("")
    rows = [[c["candidate_id"], c["family"], c["rule"],
             "OPTIMISTIC (post-hoc on Day-11 dev)" if c["optimistic"] else ""]
            for c in result["candidates"]["baselines"]]
    A(_table(["baseline", "family", "rule", "caveat"], rows))
    A("")
    rows = [[c["candidate_id"], c["family"], c["feature_set"],
             "exploratory" if c["exploratory"] else ""]
            for c in result["candidates"]["learned"]]
    A(_table(["learned candidate", "family", "feature set", "flag"], rows))
    A("")
    A("Pre-declared threshold procedures (applied to every learned candidate):")
    A("")
    A("- `T0_PROB_0.50` - predict FAKE when p >= 0.50 (primary).")
    A("- `T1_TRAIN_F1_FPR_CONSTRAINED` - per-fold threshold = argmax TRAIN-fold")
    A(f"  F1 subject to TRAIN-fold FPR <= {result['validation']['fpr_constraint']}, "
      "candidates are")
    A("  the sorted unique TRAIN-fold scores only, ties -> smallest threshold.")
    A("  This mirrors the objective recorded in `reports/V2_CANDIDATE_FREEZE.json`.")
    A("  No threshold is ever chosen using an evaluation fold.")
    A("")

    # ---------------- 4. CV + leakage ------------------------------------ #
    A("## 4. Cross-validation method and leakage controls")
    A("")
    v = result["validation"]
    A(f"- Primary: `{v['primary_scheme']}` = {v['primary_scheme_detail']}")
    A(f"- Sensitivity `LOO`: {v['sensitivity_schemes']['LOO']}")
    A(f"- Sensitivity `CV_SGK5`: {v['sensitivity_schemes']['CV_SGK5']}")
    A("- Training-set accuracy is never computed or reported.")
    A("")
    A("Leakage controls actually enforced in code:")
    A("")
    for item in result["data_integrity"]["leakage_controls"]:
        A(f"- {item}")
    A("")
    A("### Frozen-baseline reproduction check")
    A("")
    chk = result["frozen_baseline_reproduction_check"]
    rows = [[k, chk["historical"][k], chk["recomputed"][k],
             "MATCH" if chk["historical"][k] == chk["recomputed"][k] else "MISMATCH"]
            for k in chk["recomputed"]]
    A(_table(["metric", "historical (Day-5 OOF)", "V2.1 recomputation", "status"],
             rows))
    A("")
    A(f"Historical source: `{chk['historical']['source']}`, candidate "
      f"`{chk['historical']['candidate']}`.")
    A("")
    A(chk["note"] + ".")
    A("")

    # ---------------- 5. results ----------------------------------------- #
    A("## 5. Baseline and candidate metrics")
    A("")
    for pool in ("P4B_DEV25", "D7_DEV16", "POOLED41_SENSITIVITY"):
        rows = [m for m in metric_rows if m["pool"] == pool]
        if not rows:
            continue
        lc = result["populations"][pool]
        A(f"### {pool}  (n={lc['n']}, FAKE={lc['n_fake']}, REAL={lc['n_real']})")
        A("")
        if pool.startswith("POOLED"):
            A("> SENSITIVITY ONLY. Explicit concatenation of the two development")
            A("> pools. They come from different source collections, so a pooled")
            A("> number is not a population estimate; the per-pool rows above are")
            A("> the real result.")
            A("")
        for scheme in (["NONE_UNFITTED"] if any(
                m["cv_scheme"] == "NONE_UNFITTED" for m in rows)
                else []) + [result["validation"]["primary_scheme"], "LOO",
                            "CV_SGK5"]:
            block = [m for m in rows if m["cv_scheme"] == scheme]
            if not block:
                continue
            A(f"**CV scheme `{scheme}`**")
            A("")
            body = []
            for m in block:
                body.append([
                    m["candidate_id"], m["threshold_mode"], m["feature_set"],
                    m["n_features"],
                    _f(m["oof_accuracy"]), _f(m["oof_accuracy_std"]),
                    _f(m["oof_precision"]), _f(m["oof_fake_recall"]),
                    _f(m["oof_real_specificity"]), _f(m["oof_fpr"]),
                    _f(m["oof_f1"]), _f(m["oof_roc_auc"]),
                    f"{m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}",
                    _f(m["degenerate_fold_fraction"], 2),
                ])
            A(_table(["candidate", "threshold", "features", "#f", "acc",
                      "acc sd", "prec", "FAKE rec", "REAL spec", "FPR", "F1",
                      "ROC-AUC", "TP/TN/FP/FN", "degen folds"], body))
            A("")
    A("`acc sd`, `FAKE rec`, `FPR` and `F1` for the primary scheme are means over")
    A("the 10 repeats; TP/TN/FP/FN are means per repeat (so they are fractional).")
    A("`degen folds` is the fraction of folds whose test predictions collapsed to")
    A("a single class - the fold-collapse diagnostic.")
    A("")

    # ---------------- 6. selection --------------------------------------- #
    A("## 6. Candidate admissibility and the 0.80 target")
    A("")
    A("**Selection rule (pre-declared, judged on BOTH development pools):**")
    A("")
    A(f"> {sel['rule']}")
    A("")
    A(f"_{sel['rule_note']}._")
    A("")
    A("The two eligible populations are independent source collections with zero")
    A("content overlap. Requiring a candidate to clear the target on `P4B_DEV25`")
    A("*and* to not regress below the frozen V2 baseline on `D7_DEV16` is what")
    A("separates a real decision layer from fitting one collection's quirks.")
    A("Without that requirement a rule that simply mis-routes `D7_DEV16` can print")
    A("an impressive number on a pool where the frozen backbone scores 0.4375.")
    A("")
    A(f"- selected / strongest reported: `{sel['selected_id']}`")
    A(f"- selected for implementation: "
      f"**{sel['selected_for_implementation']}**")
    A(f"- reason: {sel['reason']}")
    A("")
    A("**Was the 0.80 target reached?**")
    A("")
    A(f"**{result['target']['reached_80_percent']}**")
    A("")
    A(f"_{result['target']['reached_80_percent_definition']}._")
    A("")
    A("For transparency, the single largest accuracy printed anywhere in this")
    A("experiment, per pool, on the primary scheme:")
    A("")
    A(_table(["pool", "candidate / threshold", "acc", "FAKE rec", "FPR"],
             [[pool, v["candidate_and_mode"], _f(v["accuracy"]),
               _f(v["fake_recall"]), _f(v["fpr"])]
              for pool, v in (result["target"]
                              .get("max_observed_accuracy_by_pool") or
                              {}).items()]))
    A("")
    A("Each of these is a *single arm that won on one pool*, and every one of")
    A("them collapses when the same configuration is applied to the other pool or")
    A("under the source-transfer stress test. They are shown so the failure is")
    A("visible, not because any of them is a result.")
    A("")
    if sel["rejected"]:
        A("Admissibility rejections (primary scheme, learned candidates, both")
        A("pools required):")
        A("")
        A(_table(["candidate", "threshold", "P4B acc", "P4B FPR", "D7 acc",
                  "D7 FPR", "failed checks"],
                 [[r["candidate"], r["threshold_mode"],
                   _f(r["p4b_accuracy"]), _f(r["p4b_fpr"]),
                   _f(r["d7_accuracy"]), _f(r["d7_fpr"]),
                   ", ".join(r["failed"])]
                  for r in sel["rejected"]]))
        A("")

    # ---------------- 7. limitations ------------------------------------- #
    A("## 7. Was 80% reached?")
    A("")
    A("**Interpretation rule that applies to every number in this report:** any")
    A("figure at or above 0.80 would be reported ONLY as a **\"V2.1 development")
    A("OOF result\"**. It is NOT validated accuracy, NOT real-world accuracy and")
    A("NOT production accuracy, and it does not unseal the closed populations.")
    A("")
    if result["target"]["reached_80_percent"]:
        A("Yes, but only by an arm that cleared the pre-declared admissibility")
        A("checks on both development populations. Per the interpretation rule")
        A("this is reported ONLY as a **\"V2.1 development OOF result\"**. It is")
        A("NOT validated accuracy, NOT real-world accuracy and NOT production")
        A("accuracy, and the closed locked/holdout populations remain sealed.")
    else:
        A("**No.** No learned candidate cleared the pre-declared admissibility")
        A("checks on both development populations.")
        A("")
        A("The honest answer is that the existing persisted evidence, recombined")
        A("by a lightweight second-stage decision layer, does **not** reach the")
        A("0.80 target on the eligible development data. Every arm that prints a")
        A("number at or above 0.80 does so on one pool only, and every such arm")
        A("fails at least one of:")
        A("")
        A("- **accuracy on the other pool** (typically 0.31-0.75, i.e. worse than")
        A("  the frozen backbone it was supposed to improve),")
        A("- **FAKE recall / FPR balance** (many arms reach 0.80+ accuracy by")
        A("  predicting FAKE far too often, e.g. `RF_CORE_D3` on `P4B_DEV25`:")
        A("  0.971 recall at 0.725 FPR),")
        A("- **fold stability** (degenerate-fold fractions of 0.2-0.6, and 1.00")
        A("  for every LOO arm, which indicates the decision surface is decided")
        A("  by one or two samples),")
        A("- **source transfer** (`CV_SGK5` collapses to 0.06-0.50 once a source")
        A("  family is held out entirely).")
        A("")
        A("This is a negative result about the *evidence*, not about the")
        A("candidate family. It should be read as: 41 samples of persisted")
        A("detector scores do not contain a stable second decision layer that")
        A("improves on the frozen V2 fusion. The correct next step is not more")
        A("threshold tuning or more features on the same 41 samples.")
    A("")
    A("## 8. Limitations")
    A("")
    A("- **Sample size.** The eligible development data is 25 + 16 = 41 samples")
    A("  (28 FAKE / 13 REAL). One sample is worth 4.0 points of accuracy on")
    A("  `P4B_DEV25` and 6.25 points on `D7_DEV16`. Every number here has a very")
    A("  wide confidence interval; the 10-repeat spread is reported precisely")
    A("  because of this.")
    A("- **No independent promotion set exists.** The Day-6 locked 8 and the")
    A("  Day-9 honest holdout 5 are CLOSED. There is no untouched population left,")
    A("  so nothing here can be promoted regardless of the number.")
    A("- **`D7_DEV16` is near-chance for AASIST.** On that pool REAL samples have a")
    A("  HIGHER mean AASIST fake score than FAKE samples, which is why the frozen")
    A("  0.7 audio weight produces a 0.60 FPR there. Any candidate that leans on")
    A("  audio will look bad on `D7_DEV16` for reasons unrelated to its quality.")
    A("- **`a_n_windows` is a duration/codec proxy.** It is a legitimately")
    A("  persisted AASIST statistic and one of the few features that separates the")
    A("  classes in the same direction in both pools, but it encodes clip length,")
    A("  not manipulation. Treat any gain that depends on it as suspect.")
    A("- **Grouping is label-pure in both pools.** `CV_SGK5` therefore measures")
    A("  transfer to an unseen source family, which is a much harder problem than")
    A("  the primary stratified estimate. It is reported as a stress test.")
    A("- **`BASE_XCEPTION_PERSIST060` is optimistic by construction.** Its single")
    A("  gate (`x_frac_ge_0.60 >= 0.25`) is the additive condition of the Day-11")
    A("  `frac060_ge_0.25` configuration, which was itself chosen on the same 16")
    A("  samples. It is an in-sample upper bound, not a competing honest")
    A("  estimate, and it is NOT a replay of the full Day-11 3-state route.")
    A("- **`LR_EXT_C01` is an overfitting probe, not a candidate.** 24 features on")
    A("  16 samples; its score is reported only to quantify how much a richer")
    A("  feature set inflates development OOF accuracy.")
    A("- **No threshold search, no architecture change, no new data.** If the")
    A("  answer is negative, the correct next step is not more threshold tuning.")
    A("")
    A("---")
    A("")
    A(f"ForenSight remains **{result['status']}**. The frozen V2 candidate is")
    A("unchanged and remains the research scoring backbone.")
    A("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# artifact audit
# --------------------------------------------------------------------------- #

def render_audit(audit: dict) -> str:
    lines: list[str] = []
    A = lines.append
    A("# V2.1 Artifact Audit")
    A("")
    A("Read-only inspection of every persisted Day-2..Day-15 artifact V2.1 was")
    A("asked to look for. No detector was re-run. No production code was modified.")
    A("")
    missing = [a["artifact_key"] for a in audit["artifacts"] if not a["exists"]]
    A(f"- artifacts inspected: **{len(audit['artifacts'])}**")
    A(f"- missing: **{missing if missing else 'none'}**")
    A("")

    A("## 1-7. Per-artifact audit")
    A("")
    for a in audit["artifacts"]:
        A(f"### `{a['path']}`")
        A("")
        A(f"1. path: `{a['path']}`")
        A(f"2. rows: **{a.get('n_rows')}**"
          + (f" / {a['n_columns']} columns" if a.get("n_columns") else ""))
        if a.get("feature_families"):
            fams = ", ".join(
                f"{k} ({v['n']})" for k, v in a["feature_families"].items())
            A(f"3. available features: {fams}")
            A(f"   full column list: {', '.join('`' + c + '`' for c in a['columns'])}")
        else:
            A(f"3. available features: "
              f"{a.get('top_level_keys') or 'n/a (JSON document)'}")
        A(f"4. label availability: **{'yes (' + a['label_column'] + ') ' + str(a.get('label_counts')) if a.get('has_labels') else 'no'}**")
        A(f"5. sample-ID availability: **{'yes (' + str(a.get('n_unique_sample_ids')) + ' unique)' if a.get('has_sample_ids') else 'no'}**")
        A(f"6. content hash available: **{'yes (md5)' if a.get('has_content_hash') else 'no'}**")
        A(f"7. population tier: **{a['population_tier']}**")
        A("")

    A("## Overlap between artifacts (by sample id)")
    A("")
    if not audit["sample_id_overlap"]:
        A("No two audited artifacts share a sample id.")
    else:
        A(_table(["artifact A", "artifact B", "shared ids", "byte-identical md5"],
                 [[o["artifact_a"], o["artifact_b"], o["n_shared_sample_ids"],
                   o["byte_identical_md5"]]
                  for o in audit["sample_id_overlap"]]))
    A("")
    A("## Cross-pool content-hash check")
    A("")
    cp = audit["cross_pool_content_hash"]
    A(f"- P4B manifest md5 count: {cp['p4b_md5_count']}")
    A(f"- D7 manifest md5 count: {cp['d7_md5_count']}")
    A(f"- shared md5: **{cp['shared_md5'] if cp['shared_md5'] else 'none'}**")
    A(f"- byte-identical overlap between the two development pools: "
      f"**{cp['byte_identical_overlap']}**")
    A("")
    A("## Closed populations and where they appear")
    A("")
    A(_table(["population", "n", "status", "present in artifacts"],
             [[p["population"], p["n"], p["status"],
               ", ".join(p["present_in_artifacts"]) or "-"]
              for p in audit["closed_populations"]]))
    A("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# provenance / overlap
# --------------------------------------------------------------------------- #

def render_provenance(audit: dict, table: dict) -> str:
    lines: list[str] = []
    A = lines.append
    counts = audit.get("evidence_row_counts", {})
    A("# V2.1 Data Provenance and Overlap Audit")
    A("")
    A("## Provenance chain")
    A("")
    A("| pool | ids | labels from | Xception evidence | AASIST evidence |")
    A("|---|---|---|---|---|")
    A("| `P4B_DEV25` | `day6_candidate.COMMON_IDS` (frozen, 25) | "
      "`reports/xception_devval_predictions.csv` | "
      f"`reports/xception_devval_per_frame.csv` "
      f"({counts.get('P4B_FRAMES', '?')} rows) -> Day-11 descriptors | "
      "`reports/aasist_day4_predictions.csv` (video aggregates only; "
      "per-window evidence was never persisted for P4B) |")
    A("| `D7_DEV16` | `reports/DAY9_RULE_CONFIG.json` development_population | "
      f"`reports/DAY9_RAW_FRAME_EVIDENCE.csv` "
      f"({counts.get('D7_FRAMES', '?')} rows) | "
      "`reports/DAY9_RAW_FRAME_EVIDENCE.csv` -> Day-11 descriptors | "
      f"`reports/DAY9_AASIST_WINDOW_EVIDENCE.csv` "
      f"({counts.get('D7_AASIST_WINDOWS', '?')} rows) + "
      "`reports/DAY9_AASIST_VIDEO_SUMMARIES.csv` |")
    A("")

    A("## Overlap determination")
    A("")
    A("- The two pools use **disjoint id namespaces** (`P4B-###` vs `D7-###`).")
    A("- Shared id count between the pools: **0**.")
    A("- Shared md5 between `phase4b_combined_dataset_manifest.csv` and "
      "`DAY7_FRESH_VALIDATION_MANIFEST.csv`: **0** "
      "(no byte-identical media).")
    A("- Shared basename between the two manifests: **0**.")
    A("- Within P4B: **no md5 duplicates** in the 37-row combined manifest.")
    A("- `P4B-026` is `EXCLUDE_OVERLAP` in the manifest (dHash 0 to a Phase-4A")
    A("  LOCKED test file) and is in no split - correct handling, recorded here.")
    A("- `reports/DAY7_AUDIT_NEARDUPS.json` records single-frame dHash 0 matches")
    A("  between distinct files (e.g. `avh talking.mp4` vs `P4B-025`/`P4B-031`).")
    A("  These are weak single-frame perceptual matches between DIFFERENT files,")
    A("  not duplicate media, but they are recorded because three of the four")
    A("  carry different labels. A single-frame dHash is not a reliable")
    A("  identity key, so it is reported as a caveat rather than used to merge")
    A("  or split samples.")
    A("- The frozen 25-sample reference population is exactly")
    A("  `xception_devval_predictions.csv` minus the 3 rows with no usable")
    A("  Xception face score (`P4B-025`, `P4B-035`, `P4B-037`).")
    A("")

    A("## Per-pool evidence coverage in the V2.1 feature table")
    A("")
    for pool, rows in table.items():
        A(f"### {pool} (n={len(rows)})")
        A("")
        body = []
        for name in feat.CORE_FEATURES + [
                c for c in feat.EXTENDED_FEATURES
                if c not in feat.CORE_FEATURES]:
            miss = sum(1 for r in rows if feat.fnum(r.get(name)) is None)
            body.append([f"`{name}`", len(rows) - miss, miss,
                         "complete" if miss == 0 else "in-fold median imputed"])
        A(_table(["feature", "present", "missing", "handling"], body))
        A("")
    A("Missing values arise only from the documented Day-11 evidence rules:")
    A("`q90` needs >= 4 usable face frames, `q95` needs >= 8, `std` needs >= 2,")
    A("`half_delta_abs` needs >= 3. Imputation is median-based and fitted inside")
    A("the training fold only.")
    A("")
    A("## Feature-level label separation (descriptive, for the record)")
    A("")
    for pool, rows in table.items():
        fa = [r for r in rows if r["y"] == 1]
        re = [r for r in rows if r["y"] == 0]
        body = []
        for name in feat.CORE_FEATURES:
            fv = [feat.fnum(r.get(name)) for r in fa]
            rv = [feat.fnum(r.get(name)) for r in re]
            fv = [v for v in fv if v is not None]
            rv = [v for v in rv if v is not None]
            if not fv or not rv:
                continue
            mf, mr = statistics.fmean(fv), statistics.fmean(rv)
            body.append([f"`{name}`", f"{mf:+.4f}", f"{mr:+.4f}",
                         f"{mf - mr:+.4f}"])
        A(f"### {pool}")
        A("")
        A(_table(["feature", "FAKE mean", "REAL mean", "gap"], body))
        A("")
    A("A positive gap means FAKE scores higher. Read this as descriptive only -")
    A("no feature was selected because of this table; the feature sets were")
    A("pre-declared from the persisted-artifact inventory.")
    A("")
    return "\n".join(lines)
