# Adaptive Frame Selection for Efficient Video Understanding

Select a small, informative, non-redundant set of frames from a long video **before** sending them to a vision-language model.

**Contributions**
1. **Two-stream importance scorer with a learned gate**: visual change (pixel diff, histogram, optical flow, MobileNet drift) and semantic content (CLIP + novelty + optional question similarity), fused by a per-frame gate and a small temporal Transformer.
2. **Downstream-driven training**: besides human scores (TVSum/SumMe), the scorer is trained on *utility labels* measured from the VLM itself (how much each frame helps answer the question).
3. **Budget-aware selection**: MMR (importance minus redundancy) plus a **predicted per-video frame budget** K^ (static video -> few frames, busy video -> more).

## Folder structure
```
adaptive-frames/
├── README.md
├── requirements.txt
├── config.py                  # model ids, fps, K range, dims
├── features/
│   ├── video_io.py            # decode / sample / read frames
│   ├── visual_change.py       # pixel diff, hist, Farneback flow, MobileNetV3 drift
│   ├── clip_features.py       # CLIP image + text encoder
│   ├── extract.py             # per-video feature cache (.npz) + timing
│   ├── extract_all.py         # CLI: cache a whole folder
│   └── inputs.py              # raw features -> model inputs, budget pseudo-label
├── data/
│   ├── datasets.py            # TVSum/SumMe loaders, Dataset, collate, CV folds
│   └── videoqa.py             # NExT-QA csv / jsonl loader
├── models/
│   ├── scorer.py              # two-stream gated Transformer + budget head
│   └── losses.py              # MSE + rank + diversity + budget losses
├── selection/
│   ├── mmr.py                 # greedy MMR selection
│   └── baselines.py           # uniform, random, frame-diff, k-means, CLIP-query + dispatcher
├── vlm/
│   └── qwen_runner.py         # Qwen2-VL multiple-choice QA on selected frames
├── utils/
│   ├── metrics.py             # Kendall / Spearman / top-15% F1
│   └── viz.py                 # importance curve, gate plot
├── train.py                   # training (CV or full)
├── make_utility_labels.py     # VLM-driven frame utility labels
├── eval_summarization.py      # TVSum/SumMe results table
├── eval_downstream.py         # QA accuracy vs frames/latency + Pareto plot
├── app.py                     # Gradio demo
└── scripts/run_all.sh         # whole pipeline + ablation commands
```
Create `datasets/` (your raw data) next to these; `cache/`, `checkpoints/`, `results/` are created automatically.

## Setup (Colab: set runtime to GPU)
```bash
pip install -r requirements.txt
```
Always run commands from the project root.

## Data
- **TVSum**: videos in `datasets/tvsum/video/`, annotations `datasets/tvsum/ydata-anno.tsv`.
- **SumMe** (optional): `datasets/summe/GT/*.mat`, videos in `datasets/summe/videos/`.
- **NExT-QA** (downstream): `train.csv`, `val.csv`, and the videos in `datasets/nextqa/videos/`
  (nested folders are fine; if the csv ids differ from file names, pass `--video_map map_vid_vidorID.json`).
  Use a subset (300-600 questions) on free Colab.

## Run order
```bash
python -m features.extract_all --video_dir datasets/tvsum/video --out_dir cache/tvsum --fps 1
python train.py --name tvsum_full_model --train_sets tvsum --mode cv
python eval_summarization.py --dataset tvsum --ckpt_dir checkpoints/tvsum_full_model
python make_utility_labels.py --qa_csv datasets/nextqa/train.csv --video_dir datasets/nextqa/videos --max_items 600
python train.py --name final --train_sets tvsum,qa --mode full
python eval_downstream.py --ckpt checkpoints/final/model.pt --qa_csv datasets/nextqa/val.csv --video_dir datasets/nextqa/videos
python app.py --ckpt checkpoints/final/model.pt --share
```
All ablations are in `scripts/run_all.sh` (vis-only, sem-only, fixed gate, no diversity, no budget head, lambda=1.0 i.e. no MMR via the `ours_nodiv` method).

## Notes to state honestly in your report
- The budget label is a **pseudo-label** (coverage radius `--eps`); tune `--eps` so K* spreads over ~4-24.
- TVSum summarization is evaluated with rank correlation (Otani et al. 2019) and top-15% F1, not the old knapsack F-score.
- Compute accounting in `eval_downstream.py` includes feature extraction cost per method (uniform/random = 0) plus scorer and VLM time; video decoding is excluded for all methods.
- Utility mode `single` (frame alone) is more stable for a 2B VLM; `loo` is available via `--mode loo`.
