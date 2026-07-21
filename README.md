# ForenSight
**Multi-modal deepfake & media integrity detection system**

> *Forensics + Foresight + Sight — built as a final-year project.*

---

## Architecture

```
suspicious_video.mp4
        │
        ▼
  ┌─────────────┐
  │  FFmpeg      │  demux → frames (JPEG) + audio (WAV 16 kHz mono)
  └──────┬───────┘
         │
   ┌─────┴──────┐
   │            │
   ▼            ▼
Video path   Audio path
MTCNN        STFT → Mel
   │         spectrogram
   ▼            │
EfficientNetV2  ▼
(FF++)       RawNet2
             (ASVspoof19)
   │            │
   └────┬───────┘
        ▼
   Late Fusion
   (calibrated weighted avg)
        │
        ▼
   SQLite DB  +  Streamlit Dashboard
```

## Quickstart

```bash
# 1. Clone & create venv
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model weights in models/weights/
#    - efficientnetv2_ff++.pt   (fine-tuned on FaceForensics++)
#    - rawnet2_asvspoof19.pt    (trained on ASVspoof 2019)

# 4. Run on a video
python main.py --input path/to/video.mp4 --fps 2

# 5. Launch dashboard
streamlit run dashboard/app.py
```

## Configuration

All settings live in `config/config.yaml`. Key knobs:

| Setting | Default | Range |
|---|---|---|
| `video.frame_rate` | 1 fps | 1–5 |
| `video.face_detection.no_face_strategy` | `skip` | `skip`, `flag`, `treat_as_signal` |
| `fusion.weights.video` | 0.6 | 0–1 |
| `fusion.verdict_threshold` | 0.5 | 0–1 |

## Project Structure

```
forensight/
├── config/           # config.yaml
├── data/             # raw / processed / samples (gitignored)
├── database/         # SQLite DAL (db.py)
├── models/           # weights + checkpoints (gitignored)
├── pipeline/
│   ├── video/        # MTCNN + EfficientNetV2 + GradCAM
│   ├── audio/        # STFT + RawNet2
│   └── fusion/       # late fusion + calibration
├── dashboard/        # Streamlit app
├── utils/            # config, logger, demux, types
├── tests/            # pytest unit + integration
├── scripts/          # training / eval helpers
└── main.py           # CLI orchestrator
```

## Datasets

| Dataset | Used for |
|---|---|
| FaceForensics++ | EfficientNetV2 fine-tuning |
| ASVspoof 2019 | RawNet2 training |
| FakeAVCeleb / LAV-DF | Multimodal test evaluation |

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```
