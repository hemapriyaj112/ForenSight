#!/usr/bin/env python3
"""Download the official GenConViT pretrained inference weights.

WHY THIS SCRIPT EXISTS
----------------------
Phase 2A of the ForenSight audit integrates GenConViT as a real, free, locally
runnable video deepfake detector. This script fetches the official pretrained
checkpoints published by the GenConViT authors so they live in a clear local
directory (``models/genconvit/``) instead of being silently downloaded inside
application startup.

SOURCE / PROVENANCE (verify before trusting)
-------------------------------------------
  * Model weights repository : https://huggingface.co/Deressa/GenConViT
  * Reference implementation  : https://github.com/erprogs/GenConViT
  * Paper                     : Wodajo et al., "Deepfake Video Detection Using
                                Generative Convolutional Vision Transformer"
                                (arXiv:2307.07036, MDPI Appl. Sci. 2025)

LICENSE
-------
The GenConViT code and the released inference weights are distributed under the
**MIT License** (see https://github.com/erprogs/GenConViT/blob/main/LICENSE).
MIT is a permissive open-source license; commercial and research use is allowed
provided the copyright notice is retained. This satisfies ForenSight's
"prefer permissive/open-source licensing" constraint. No paid API, no cloud
inference, no account required.

WHAT IS DOWNLOADED
------------------
Two state-dict-only checkpoints (per the upstream README these are roughly half
the size of a fully-trained checkpoint because only the inference graph weights
are included):

  * genconvit_ed_inference.pth   -- Autoencoder (ED) network
  * genconvit_vae_inference.pth  -- Variational Autoencoder (VAE) network

The combined "genconvit" detector runs BOTH networks and averages their
per-frame fake probabilities (see pipeline/video/genconvit/).

REPRODUCIBILITY / SAFETY
------------------------
  * Files are downloaded ONCE. If a file already exists and is larger than a
    small sanity threshold it is NOT re-downloaded (no silent re-fetch at app
    launch; this is a manual, auditable step).
  * After download we verify each file exists and is non-trivial in size.
  * We do NOT run any model code here — this script only moves bytes.
  * To force a re-download delete the file under models/genconvit/ first.

USAGE
-----
    py scripts/download_genconvit_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ID = "Deressa/GenConViT"
HERE = Path(__file__).resolve().parent
TARGET_DIR = (HERE / ".." / "models" / "genconvit").resolve()

# Filenames exactly as published on the Hugging Face repo.
FILES = (
    "genconvit_ed_inference.pth",
    "genconvit_vae_inference.pth",
)

# Exact byte sizes (from the Hugging Face CDN Content-Length header). Used so a
# slow/truncated partial download is NEVER mistaken for a complete file — a
# partial VAE weight (2.78 GB) must be re-fetched, not skipped.
EXPECTED_SIZES = {
    "genconvit_ed_inference.pth": 238_174_949,
    "genconvit_vae_inference.pth": 2_782_175_601,
}

# A checkpoint smaller than this is almost certainly a truncated/failed
# download. The ED and VAE inference weights are each hundreds of MB to a few GB.
MIN_BYTES = 50 * 1024 * 1024  # 50 MB sanity floor

# huggingface_hub resumes interrupted downloads automatically (it keeps an
# .incomplete blob in its cache and continues). So a re-run after a timeout
# keeps going from where it left off.


def _download_via_hf_hub(filename: str, target: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "huggingface_hub is required to download GenConViT weights "
            "(it ships with transformers). Import failed: %s" % exc
        )
    # local_dir_use_symlinks=False copies the real file into TARGET_DIR so the
    # weights are self-contained and not dependent on the HF cache symlink.
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
    )
    # hf_hub_download may place it under TARGET_DIR/<filename> or a subpath;
    # ensure the canonical name exists.
    dest = TARGET_DIR / filename
    if Path(path).resolve() != dest.resolve():
        import shutil

        shutil.copyfile(path, dest)


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print("GenConViT weight downloader")
    print("  source : https://huggingface.co/%s" % REPO_ID)
    print("  license: MIT")
    print("  target : %s" % TARGET_DIR)
    print()

    ok = True
    for fname in FILES:
        dest = TARGET_DIR / fname
        expected = EXPECTED_SIZES.get(fname)
        if (
            dest.exists()
            and expected is not None
            and dest.stat().st_size == expected
        ):
            mb = dest.stat().st_size / (1024 * 1024)
            print("[skip] %s already present (%.1f MB, exact size)" % (fname, mb))
            continue
        if dest.exists():
            cur = dest.stat().st_size
            print(
                "[get ] %s ... (resuming, have %.1f MB of %.1f MB)"
                % (fname, cur / (1024 * 1024), (expected or 0) / (1024 * 1024))
            )
        else:
            print("[get ] %s ..." % fname)
        try:
            _download_via_hf_hub(fname, dest)
        except Exception as exc:
            print("  ERROR downloading %s: %s" % (fname, exc), file=sys.stderr)
            ok = False
            continue
        if not dest.exists() or dest.stat().st_size < MIN_BYTES or (
            expected is not None and dest.stat().st_size != expected
        ):
            print(
                "  ERROR: %s missing, too small, or size mismatch after "
                "download (expected %s bytes, got %s) -- possible truncated "
                "transfer"
                % (
                    fname,
                    expected,
                    dest.stat().st_size if dest.exists() else "absent",
                ),
                file=sys.stderr,
            )
            ok = False
            continue
        mb = dest.stat().st_size / (1024 * 1024)
        print("  done  %s (%.1f MB)" % (fname, mb))

    print()
    if ok:
        print("All GenConViT weights present in: %s" % TARGET_DIR)
        print("Next step: run the ForenSight pipeline / evaluation script.")
        return 0
    print(
        "One or more downloads failed. Check network access to huggingface.co "
        "and re-run this script.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
