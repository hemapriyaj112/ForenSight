#!/usr/bin/env python3
"""Download the official AASIST pretrained audio anti-spoofing checkpoint.

WHY THIS SCRIPT EXISTS
----------------------
Phase 2B of the ForenSight audit integrates AASIST as a real, free, locally
runnable audio anti-spoofing detector. This script fetches the OFFICIAL
pretrained checkpoint published by the AASIST authors so it lives in a clear
local directory (``models/aasist/``) instead of being silently downloaded at
application startup.

SOURCE / PROVENANCE (verify before trusting)
--------------------------------------------
  * Repository   : https://github.com/clovaai/aasist  (commit a04c9863)
  * Checkpoint   : committed ON-DISK in that repository at
                   models/weights/AASIST.pth — the exact weights the authors'
                   README evaluates (EER 0.83%, min t-DCF 0.0275 on
                   ASVspoof2019 LA). This is the authors' own checkpoint, not
                   a third-party mirror.
  * Paper        : Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
                   Spectro-Temporal Graph Attention Networks" (arXiv:2110.01200)
  * Architecture : `Model` in pipeline/aasist/model.py (verbatim upstream).

LICENSE
-------
MIT (Copyright (c) 2021-present NAVER Corp.) — see pipeline/aasist/model.py
header and the LICENSE/NOTICE copied under reports/PHASE2B_AASIST_EVALUATION.md.
MIT is a permissive open-source license; commercial and research use is allowed
provided the copyright notice is retained. This satisfies ForenSight's
"prefer permissive/open-source licensing" constraint. No paid API, no cloud
inference, no account required.

REPRODUCIBILITY / SAFETY
------------------------
  * Files are downloaded ONCE. If models/aasist/AASIST.pth already exists and
    its size equals the exact upstream byte size it is NOT re-downloaded (no
    silent re-fetch at app launch; this is a manual, auditable step).
  * The transfer goes to a ``.part`` file first and is only renamed into place
    when the exact expected size is verified, so a slow/truncated download is
    never mistaken for a complete checkpoint.
  * We do NOT run any model code here — this script only moves bytes.

USAGE
-----
    py scripts/download_aasist_weights.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAW_URL = (
    "https://raw.githubusercontent.com/clovaai/aasist/"
    "main/models/weights/AASIST.pth"
)
REPO_URL = "https://github.com/clovaai/aasist"
COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"

HERE = Path(__file__).resolve().parent
TARGET_DIR = (HERE / ".." / "models" / "aasist").resolve()
FILE_NAME = "AASIST.pth"

# Exact byte size of the official checkpoint on GitHub (Content-Length). Used
# so a truncated transfer is ALWAYS re-fetched, never skipped.
EXPECTED_BYTES = 1_281_532


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    dest = TARGET_DIR / FILE_NAME
    part = dest.with_suffix(".part")

    print("AASIST weight downloader")
    print("  source : %s/raw/%s/models/weights/%s" % (REPO_URL, COMMIT, FILE_NAME))
    print("  license: MIT (Copyright (c) 2021-present NAVER Corp.)")
    print("  target : %s" % dest)
    print()

    if dest.exists() and dest.stat().st_size == EXPECTED_BYTES:
        mb = dest.stat().st_size / (1024 * 1024)
        print("[skip] %s already present (%.1f MB, exact size)" % (FILE_NAME, mb))
        print("Next step: run the ForenSight pipeline / evaluation script.")
        return 0

    if part.exists():
        print("[get ] resuming/overwriting %s.part (%.1f MB so far)"
              % (FILE_NAME, part.stat().st_size / (1024 * 1024)))
    else:
        print("[get ] %s ..." % FILE_NAME)

    try:
        urllib.request.urlretrieve(RAW_URL, str(part))  # noqa: S310 - pinned https
    except Exception as exc:  # pragma: no cover - network/environment dependent
        print("  ERROR downloading %s: %s" % (FILE_NAME, exc), file=sys.stderr)
        print("  If GitHub raw is blocked, download the file manually from")
        print("  https://github.com/clovaai/aasist/blob/main/models/weights/AASIST.pth")
        print("  and place it at %s" % dest, file=sys.stderr)
        return 1

    if not part.exists() or part.stat().st_size != EXPECTED_BYTES:
        got = part.stat().st_size if part.exists() else 0
        print(
            "  ERROR: size mismatch after download (expected %s bytes, got %s) "
            "-- possible truncated transfer"
            % (EXPECTED_BYTES, got),
            file=sys.stderr,
        )
        return 1

    part.replace(dest)
    mb = dest.stat().st_size / (1024 * 1024)
    print("  done  %s (%.1f MB)" % (FILE_NAME, mb))
    print()
    print("All AASIST weights present in: %s" % TARGET_DIR)
    print("Next step: run the ForenSight pipeline / evaluation script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())