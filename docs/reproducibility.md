# Reproducibility record

Fill in once, keep updated. Numbers below come from files/runs, never retyped from memory.

- OpenGait repo + commit: https://github.com/ShiqiYu/OpenGait @ 4d228e154ae6f7473db49e0ef33493be0c137601 (pinned 2026-10-15)
- GaitBase config file: `configs/gaitbase/gaitbase_da_casiab.yaml` (OpenGait); evaluator fp16 on, all_ordered, frames_all_limit 720
- Checkpoint file + source: `GaitBase_DA-60000.pt`, https://huggingface.co/opengait/OpenGait/resolve/main/CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt — sha256 prefix `ff16d73d7678`
- GaitSet: no released CASIA-B checkpoint; trained via `notebooks/01b_gaitset_training.ipynb` + `configs/opengait/gaitset_casiab_1gpu.yaml` → (fill: sha256 prefix)
- CASIA-B source: Kaggle mirror https://www.kaggle.com/datasets/trnquanghuyn/casia-b (534,440,514 bytes, updated 2023-04-14, licence "Unknown"; CASIA terms still apply; official: http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip). Cite Yu et al. 2006.
  - Mirror is ALREADY pretreated: 64×64 uint8 PNGs under `output/` (GaitSet `pretreatment.py` output; OpenGait's pretreatment is the same algorithm). Re-applying OpenGait's per-frame crop/resize left 3388/3390 sampled frames (40 random seqs, seed 0) bit-identical.
  - So frames are packed untouched with OpenGait's own script in its no-transform mode: `python datasets/pretreatment.py -i <data>/archive/output -o <data>/CASIA-B-pkl -d GREW -n 8` (from OpenGait dir).
  - Completeness: 13,592 / 13,640 sequence folders present, 0 empty, 1,117,083 frames. 48 missing (33 train, 15 test): 005/*/000 (all 10 types); 026/cl-02/162; 037/bg-01/{144,162,180}; 037/bg-02/{054,126,162}; 048/cl-01/162; 048/cl-02/{054,126,144,162}; 068/nm-01/{054,126,144,162}; 068/nm-02/{126,162}; 068/nm-03/{126,144,180}; 068/nm-04/{126,144}; 079/bg-02/{054,162}; 088/nm-03/{054,126,162,180}; 088/nm-04/{054,126,144,162,180}; 109/nm-01/{126,144,162,180}. Whether these match gaps in the official zip is unverified; the clean gate (within 1% of OpenGait) is the acceptance test.
- Split file: OpenGait `datasets/CASIA-B/CASIA-B.json` (train 001–074, test 075–124)
- Python / PyTorch / CUDA: (fill from results/raw/env_baseline.json)
- GPU: (fill)
- Global seed(s): 0
- Clean baseline Rank-1 (NM/BG/CL) obtained: (fill from Notebook 01 gate table)
- OpenGait reported Rank-1 (NM/BG/CL) for comparison: GaitBase 97.6 / 94.0 / 77.4; GaitSet 95.8 / 90.0 / 75.4 (OpenGait `docs/1.model_zoo.md` at the pinned commit, identical view excluded)
