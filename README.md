# DL-Project — Digital-to-Analog Clock Converter

A deep-learning pipeline that reads the time from a **digital clock image** and transfers it onto an **analog clock image** — replacing whatever time the analog clock showed with the correct one.

```
Digital clock  →  [Reader]  →  (H, M, S)  →  [Draw]  →  Analog clock showing the right time
Analog clock   →  [Eraser]  →  clean face  ↗
```

---

## How It Works

The pipeline has three stages:

| Stage | Component | What it does |
|---|---|---|
| 1 | `DigitalClockClassifier` | ResNet-18 that reads H:M:S from a digital clock image |
| 2 | `ClockEraserV2` | U-Net that removes hands from an analog clock image |
| 3 | `draw_hands_on_tensor` | Pure geometry that draws the correct hands back on |

No single model does everything — the problem is decomposed into small, well-defined pieces.

---

## Project Structure

```
DL-Project/
├── pipeline/
│   ├── full_pipeline.py      # End-to-end orchestrator (ClockPipeline + 3 run modes)
│   ├── analog_reader.py      # Model 1 — digital clock classifier (ResNet-18)
│   ├── hand_eraser.py        # Model 2 — U-Net clock hand eraser
│   └── draw_hand.py          # Geometry-based hand drawing (no model)
│
├── src/
│   ├── data_generator.py     # Synthetic dataset generator
│   ├── dataset.py            # PyTorch Dataset wrapper
│   ├── clock_dataset/        # Generated images (train/ and test/)
│   ├── checkpoints/          # Trained model weights (not tracked by git)
│   │   ├── digital_reader_best.pth
│   │   └── eraser_v2_best.pth
│   ├── hand_drawer.ipynb     # Experiments: hand drawing geometry
│   ├── read_analog.ipynb     # Experiments: analog reading
│   └── train_geometry.ipynb  # Training walkthrough
│
├── run_pipeline.ipynb        # ← START HERE to run the pipeline
├── Project_Walkthrough.ipynb # Full training walkthrough
├── guidebook.md              # Design decisions and architecture log
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python src/data_generator.py \
    --output_dir src/clock_dataset \
    --train_count 1000 \
    --test_count 200
```

This creates paired digital + analog clock images. The test set uses times **not seen during training** — no data leakage.

### 3. Train the models

Open `Project_Walkthrough.ipynb` and run all cells. Trained weights are saved to:
- `src/checkpoints/digital_reader_best.pth`
- `src/checkpoints/eraser_v2_best.pth`

### 4. Run the pipeline

Open `run_pipeline.ipynb`. Three modes are available:

| Mode | What it does |
|---|---|
| **Single** | Convert one digital + analog image pair |
| **Batch** | Evaluate 6 random test samples side-by-side |
| **Animate** | Live clock ticking at real wall-clock time |

---

## Model Details

### Model 1 — Digital Clock Reader (`DigitalClockClassifier`)

- **Input:** `(B, 3, 224, 224)` ImageNet-normalized tensor
- **Architecture:** ResNet-18 → shared bottleneck (Linear 512→256, ReLU, Dropout 0.3) → three classification heads
  - `hour_head`: 24 classes (0–23)
  - `minute_head`: 60 classes (0–59)
  - `second_head`: 60 classes (0–59)
- **Why classification, not regression?** Regression loses work poorly near boundaries (e.g. 12:59 vs 13:00). Treating each digit as a separate class avoids this entirely.

### Model 2 — Clock Hand Eraser (`ClockEraserV2`)

- **Input / Output:** `(B, 3, 256, 256)` tensor
- **Architecture:** 4-level U-Net with skip connections
  - Encoder: 4 × double-conv blocks + MaxPool (channels: 3→64→128→256→512)
  - Bottleneck: double-conv at 512 channels
  - Decoder: 4 × ConvTranspose2d upsample + skip concat + double-conv
  - Output: `Conv2d → Sigmoid` → pixel values in [0, 1]

### Hand Drawing (no model)

Angles are computed from `(H, M, S)` using continuous motion (e.g. the hour hand moves smoothly between hour markers based on minutes elapsed). A `-90°` offset places 12 at the top.

---

## Dataset

All images are **synthetically generated** — no real photographs needed.

- **Digital styles:** simple text, LCD, segmented display
- **Analog styles:** 10 colour palettes × 4 marker styles × 4 hand styles
- Each sample includes: digital image, analog image with hands, analog image without hands (ground truth for the eraser)
- `labels.csv` columns: `digital_filename, analog_filename, analog_clean_filename, hour, minute, second`

---

## Results

Outputs are saved to `results/` after each run:

- `results/output_single.png` — single conversion output
- `results/output_single_comparison.png` — side-by-side: digital in / analog in / output
- `results/batch_results.png` — 4-row grid for batch evaluation (correct reads in green, wrong in red)
