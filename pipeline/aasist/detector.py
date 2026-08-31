"""pipeline/aasist/detector.py

AASIST — a real, free, locally-runnable pretrained audio anti-spoofing detector.

This wraps the OFFICIAL AASIST implementation and its OFFICIAL pretrained
checkpoint (ASVspoof2019 LA) behind a clean interface so an audio track can be
scored with an actual neural anti-spoofing countermeasure.

SOURCE / PROVENANCE (verified)
------------------------------
  * Reference implementation : https://github.com/clovaai/aasist  (commit a04c9863)
  * Weights                  : committed directly in that repo at
                               models/weights/AASIST.pth (1,281,532 bytes) —
                               the same checkpoint the authors use for the
                               README evaluation (EER 0.83%, min t-DCF 0.0275).
  * Paper                    : Jung et al. ICASSP 2022, arXiv:2110.01200
  * License                  : MIT (Copyright (c) 2021-present NAVER Corp.) —
                               see pipeline/aasist/model.py header + LICENSE.

DESIGN
------
* The model + weights are loaded LAZILY the first time inference is requested,
  and its status is reported via `aasist_status` (ACTIVE_MODEL | INACTIVE) with
  an exact reason when inactive. There is NO silent fallback to the audio
  heuristic and NO fusion into the verdict: AASIST is evaluation-only in
  Phase 2B (config.aasist.use_in_final_fusion=false).
* Input preprocessing mirrors the upstream eval path `data_utils.pad()`
  (deterministic first-N-samples window, tile-repeat padding, 16 kHz mono). A
  whole clip is scanned with fixed non-overlapping 64600-sample (~4.04 s)
  windows so score applies to the full audio timeline, not just the first 4 s.
* Class/label semantics (verified against upstream data_utils.py + main.py):
      class 0 = SPOOF,  class 1 = BONAFIDE    (bonafide==real human speech)
  The released model's own score is the raw BONAFIDE logit (logits[:,1], higher
  = more bona fide) written by upstream main.py. We additionally expose a
  bounded 0-1 `aasist_spoof_score` = mean over windows of softmax(logits)[0]
  (higher = more SPOOF). These are NOT calibrated probabilities and must not be
  read as P("fake video"): audio spoofing != video forgery.
* All output names avoid the word "probability" (not calibrated).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Upstream config/config/AASIST.conf -> model_config (verified, immutable).
_MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "aasist"
_WEIGHT = "AASIST.pth"

# Fixed by upstream data_utils.py (eval path) + config/AASIST.conf.
_SAMPLE_RATE = 16_000
_NB_SAMP = 64_600  # ~4.0375 s @ 16 kHz

STATUS_ACTIVE_MODEL = "ACTIVE_MODEL"
STATUS_INACTIVE = "INACTIVE"

_LABEL_SPOOF = "SPOOF"
_LABEL_BONAFIDE = "BONAFIDE"

_AGGREGATION_METHOD = "mean_softmax_spoof_prob_over_64600_sample_windows"


def _pad_window(window: np.ndarray, max_len: int = _NB_SAMP) -> np.ndarray:
    """Mirror upstream data_utils.pad() for a single window.

    Truncate if long enough, else tile-repeat to max_len (deterministic —
    the SAME input always yields the same window, unlike pad_random).
    """
    x = np.asarray(window, dtype=np.float32)
    if x.size >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x.size) + 1
    return np.tile(x, num_repeats)[:max_len]


def _window_waveform(waveform: np.ndarray, max_len: int = _NB_SAMP) -> list[np.ndarray]:
    """Split a 16 kHz mono waveform into fixed non-overlapping windows.

    Every window is then run through `_pad_window` so each one obeys the
    upstream eval contract (exactly max_len samples). A silent/too-short clip
    still yields its first window (matching upstream behaviour for a short
    utterance: short clips ARE scored after tile padding upstream).
    """
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.size == 0:
        return []
    if wav.size < max_len:
        return [_pad_window(wav, max_len)]
    windows = [
        _pad_window(wav[i:i + max_len], max_len)
        for i in range(0, wav.size - max_len + 1, max_len)
    ]
    # Tail window (if any) is shorter than max_len -> tile-padded, scored.
    if wav.size % max_len != 0:
        windows.append(_pad_window(wav[wav.size - (wav.size % max_len):], max_len))
    return windows


def _label_for_spoof_score(spoof_score: float) -> str:
    """class 0 = SPOOF, class 1 = BONAFIDE (argmax over the mean 2-vector)."""
    return _LABEL_SPOOF if spoof_score >= 0.5 else _LABEL_BONAFIDE


class AASISTDetector:
    """Lazy-loading wrapper around the official AASIST weights.

    Parameters
    ----------
    weights_dir : Path, optional
        Directory containing AASIST.pth. Default <project>/models/aasist.
    window_samples : int, optional
        Fixed input length per window (upstream nb_samp, default 64600).
    sample_rate : int, optional
        Target sample rate (upstream 16000; files are resampled if needed).
    """

    def __init__(
        self,
        weights_dir: Optional[Path | str] = None,
        window_samples: int = _NB_SAMP,
        sample_rate: int = _SAMPLE_RATE,
    ):
        self._weights_dir = Path(weights_dir) if weights_dir else _DEFAULT_DIR
        self.window_samples = int(window_samples)
        self.sample_rate = int(sample_rate)

        self.available = False
        self.load_error: Optional[str] = None
        self.model = None
        self.device = "cuda" if self._torch_cuda_available() else "cpu"
        self.model_params: Optional[int] = None
        self._loaded = False

    @staticmethod
    def _torch_cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:  # pragma: no cover - environment dependent
            return False

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    @property
    def status(self) -> str:
        if self._loaded and self.available:
            return STATUS_ACTIVE_MODEL
        return STATUS_INACTIVE

    def ensure_loaded(self) -> bool:
        """Load the weights + model if not already loaded.

        Returns True if the model is ready for inference. On any failure sets
        self.available=False and self.load_error with the exact reason. Never
        raises: callers check `status`/`available` and report INACTIVE.
        """
        if self._loaded:
            return self.available
        self._loaded = True
        try:
            self._do_load()
            self.available = True
        except FileNotFoundError as exc:
            self.load_error = f"missing AASIST checkpoint: {exc}"
            self.available = False
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.available = False
        return self.available

    def _do_load(self) -> None:
        import torch

        wpath = self._weights_dir / _WEIGHT
        if not wpath.exists():
            raise FileNotFoundError(
                f"AASIST checkpoint not found at {wpath}. Run "
                "scripts/download_aasist_weights.py first."
            )

        from pipeline.aasist.model import Model

        model = Model(_MODEL_CONFIG)
        sd = torch.load(str(wpath), map_location="cpu")
        # Official weights are a plain state dict; tolerate an archive wrapper.
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(
                f"AASIST state_dict missing {len(missing)} keys during "
                f"load: {missing[:5]}"
            )
        if unexpected:
            raise RuntimeError(
                f"AASIST state_dict has {len(unexpected)} unexpected keys: "
                f"{unexpected[:5]}"
            )
        model.to(self.device).eval()
        self.model = model
        self.model_params = sum(p.numel() for p in model.parameters())

    # ------------------------------------------------------------------ #
    # Waveform loading (mirrors audio_forensics._load_waveform)
    # ------------------------------------------------------------------ #
    def _load_waveform(self, wav_path: Path) -> Optional[np.ndarray]:
        """Read a WAV as mono float32 at self.sample_rate (16 kHz).

        Returns None if the file is missing/unreadable/empty.
        """
        try:
            import soundfile as sf

            data, sr = sf.read(str(wav_path), dtype="float32")
        except Exception:  # noqa: BLE001 - never crash on a bad wav
            return None
        if data is None or data.size == 0:
            return None
        if data.ndim > 1:
            data = data.mean(axis=-1)
        data = np.asarray(data, dtype=np.float32)
        if sr != self.sample_rate and sr > 0:
            t = np.linspace(0, data.size / sr, data.size, endpoint=False)
            new_t = np.linspace(
                0, t[-1], int(round(data.size * self.sample_rate / sr)),
                endpoint=False,
            )
            data = np.interp(new_t, t, data).astype(np.float32)
        return data

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def infer_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = _SAMPLE_RATE,
    ) -> dict:
        """Score a raw 16 kHz mono waveform with AASIST.

        Returns dict with aasist_status / aasist_spoof_score /
        aasist_bonafide_score / aasist_predicted_label / per-window scores.
        On load failure status is INACTIVE with `aasist_load_error` set and
        scores None. No heuristic fallback.
        """
        if not self.ensure_loaded():
            return self._inactive_result(
                note="AASIST cannot load; no neural spoof score produced."
            )

        import torch

        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sample_rate != self.sample_rate and sample_rate > 0 and wav.size:
            t = np.linspace(0, wav.size / sample_rate, wav.size, endpoint=False)
            new_t = np.linspace(
                0, t[-1], int(round(wav.size * self.sample_rate / sample_rate)),
                endpoint=False,
            )
            wav = np.interp(new_t, t, wav).astype(np.float32)

        windows = _window_waveform(wav, self.window_samples)
        if not windows:
            return {
                "aasist_status": STATUS_ACTIVE_MODEL,
                "aasist_load_error": None,
                "aasist_spoof_score": None,
                "aasist_bonafide_score": None,
                "aasist_mean_bonafide_logit": None,
                "aasist_predicted_label": None,
                "aasist_window_count": 0,
                "aasist_window_scores": [],
                "aasist_window_start_sec": [],
                "aasist_audio_seconds": round(float(wav.size) / self.sample_rate, 3),
                "aasist_aggregation_method": _AGGREGATION_METHOD,
                "aasist_note": "Empty waveform; no windows scored.",
                "device": self.device,
                "model_params": self.model_params,
            }

        window_scores: list[float] = []
        start_sec: list[float] = []
        with torch.no_grad():
            for i, win in enumerate(windows):
                x = torch.from_numpy(win).unsqueeze(0).to(self.device)
                _, logits = self.model(x)  # logits[0]=spoof, logits[1]=bonafide
                p = torch.softmax(logits, dim=1)  # (1, 2)
                window_scores.append(float(p[0, 0].item()))
                start_sec.append(round(i * self.window_samples / self.sample_rate, 3))

        spoof_score = float(np.mean(window_scores))
        bonafide_score = 1.0 - spoof_score  # exact: 2-class softmax over a window

        return {
            "aasist_status": STATUS_ACTIVE_MODEL,
            "aasist_load_error": None,
            "aasist_spoof_score": round(spoof_score, 6),
            "aasist_bonafide_score": round(bonafide_score, 6),
            "aasist_mean_bonafide_logit": None,
            "aasist_predicted_label": _label_for_spoof_score(spoof_score),
            "aasist_window_count": len(window_scores),
            "aasist_window_scores": [round(s, 6) for s in window_scores],
            "aasist_window_start_sec": start_sec,
            "aasist_audio_seconds": round(float(wav.size) / self.sample_rate, 3),
            "aasist_aggregation_method": _AGGREGATION_METHOD,
            "aasist_note": None,
            "device": self.device,
            "model_params": self.model_params,
        }

    def infer_wav(self, wav_path: Path | str | None) -> dict:
        """Score a WAV file (or None when the source video has no audio track).

        * wav_path is None        -> INACTIVE, score None, note "No audio track".
        * wav file unreadable     -> INACTIVE, score None, exact reason.
        * model/weights missing   -> INACTIVE, score None, load_error set.
        """
        if wav_path is None:
            return self._inactive_result(
                note="No audio track",
                readable=False,
            )
        wav_path = Path(wav_path)
        if not wav_path.exists():
            return self._inactive_result(
                note=f"Audio wav file not found: {wav_path}", readable=False
            )
        wav = self._load_waveform(wav_path)
        if wav is None:
            return self._inactive_result(
                note=f"Could not read / decode audio file: {wav_path}",
                readable=False,
            )
        out = self.infer_waveform(wav, sample_rate=self.sample_rate)
        if out["aasist_status"] == STATUS_INACTIVE:
            return out
        out["aasist_load_error"] = self.load_error
        return out

    def _inactive_result(self, note: str, readable: bool = True) -> dict:
        """Uniform INACTIVE result: score None, explicit note, no fallback."""
        return {
            "aasist_status": STATUS_INACTIVE,
            "aasist_load_error": self.load_error,
            "aasist_spoof_score": None,
            "aasist_bonafide_score": None,
            "aasist_mean_bonafide_logit": None,
            "aasist_predicted_label": None,
            "aasist_window_count": 0,
            "aasist_window_scores": [],
            "aasist_window_start_sec": [],
            "aasist_audio_seconds": None,
            "aasist_aggregation_method": _AGGREGATION_METHOD,
            "aasist_note": note,
            "device": self.device,
            "model_params": self.model_params,
            "aasist_weights_dir": str(self._weights_dir),
            "aasist_audio_readable": readable or False,
        }