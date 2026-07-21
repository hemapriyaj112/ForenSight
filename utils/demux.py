"""
utils/demux.py — extract video frames and audio WAV from an MP4.

Uses ffmpeg subprocess calls so there is no Python AV dependency.
Returns (frames_dir, wav_path) as Path objects.
Raises DemuxError on any failure.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from utils.logger import get_logger

log = get_logger("forensight.demux")


class DemuxError(RuntimeError):
    """Raised when ffmpeg demux fails."""


def demux(video_path: str | Path, output_dir: str | Path | None = None) -> tuple[Path, Path | None]:
    """
    Demultiplex *video_path* into frames and audio.

    Parameters
    ----------
    video_path:
        Path to the source MP4 (or any ffmpeg-readable container).
    output_dir:
        Directory to write frames/ and audio.wav into.
        A temporary directory is created if None (caller must clean up).

    Returns
    -------
    (frames_dir, wav_path)
        frames_dir — directory containing frame_NNNNNN.jpg files
        wav_path   — 16 kHz mono WAV file, or None if the source has no
                     audio stream (common for muted/silent clips, e.g.
                     WhatsApp-forwarded videos) — video-only analysis
                     still proceeds in that case rather than failing the
                     whole pipeline; see main.py/AudioDetector.run(None).

    Raises
    ------
    DemuxError only for genuine failures (corrupt/unreadable video, frame
    extraction failure). A missing audio stream is NOT treated as an error.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise DemuxError(f"Video file not found: {video_path}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="forensight_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    wav_path = output_dir / "audio.wav"

    _extract_frames(video_path, frames_dir)

    if not _has_audio_stream(video_path):
        log.info("No audio stream detected in %s — skipping audio extraction, video-only analysis will proceed", video_path)
        return frames_dir, None

    try:
        _extract_audio(video_path, wav_path)
    except DemuxError as exc:
        # Audio stream was reported present but extraction still failed
        # (corrupt/unsupported audio codec, etc.) — degrade gracefully
        # rather than losing the video analysis too, since video is the
        # primary signal and audio is a secondary corroborating one.
        log.warning("Audio extraction failed despite a detected audio stream (%s) — "
                    "continuing with video-only analysis", exc)
        return frames_dir, None

    return frames_dir, wav_path


def _has_audio_stream(video_path: Path) -> bool:
    """Quick ffprobe check for the presence of an audio stream, so we never
    attempt (and fail) an audio extraction on a video that simply doesn't
    have one."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _run(cmd: list[str], label: str) -> None:
    log.debug("demux cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DemuxError(
            f"ffmpeg {label} failed (exit {result.returncode}):\n{result.stderr}"
        )


def _extract_frames(video_path: Path, frames_dir: Path) -> None:
    """Extract 1 fps JPEG frames."""
    pattern = str(frames_dir / "frame_%06d.jpg")
    _run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "fps=1",
            "-q:v", "2",
            pattern,
        ],
        label="frame extraction",
    )
    frame_count = len(list(frames_dir.glob("*.jpg")))
    log.info("Extracted %d frames → %s", frame_count, frames_dir)


def _extract_audio(video_path: Path, wav_path: Path) -> None:
    """Extract 16 kHz mono WAV."""
    _run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            str(wav_path),
        ],
        label="audio extraction",
    )
    log.info("Extracted audio → %s", wav_path)