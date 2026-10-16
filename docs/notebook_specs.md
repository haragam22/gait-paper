# Notebook specifications

Three parallel workstreams. **Team is now two people: owners, interfaces and gates are in `docs/implementation.md`, which overrides the Owner column below.** Each notebook: Cell 1 records metadata (date, seed, platform,
GPU, Python/PyTorch/CUDA, OpenGait commit, config). Pilot on the fixed subset first. Every
notebook ends with a markdown cell: findings, limitations, next step. Long jobs follow the
compute rules in CLAUDE.md (resumable, checkpoint every ≤30 min, chunks <3 h).

| Workstream | Owner | Notebooks |
|---|---|---|
| WS1 — Foundations & controlled attacks | Teammate 1 | 01, 02, 03, 04 |
| WS2 — He et al. reproduction | Teammate 2 | 01b, 05, 06 |
| WS3 — Hou et al. reproduction & defence | Teammate 3 | 07, 11 |
| Joint — analysis | All | 08, 09, 10, 12 |

WS2 and WS3 start by reading their paper on Oct 15–16 and depend on WS1's `src/data.py`
and `src/evaluation.py` from Oct 17.

---

## WS1 — Foundations & controlled attacks

### 01_baseline_and_data_validation — due Oct 17
1. Environment check; load data from the shared Kaggle Dataset.
2. Validate: 50 test subjects; sequence counts per subject/type/view; shape 64×44;
   frame-count histogram.
3. Load GaitBase from OpenGait config + released checkpoint (GaitBase_DA-60000.pt). GaitSet
   has no released CASIA-B checkpoint: it is trained in 01b and loaded here once available.
4. Clean evaluation via `evaluate_attack(attack=None)` for both models, per condition and view.
5. Plot 3 sequences per condition; check frame order and broken frames.
6. Save `results/raw/baseline_results.csv`, `results/raw/env_baseline.json`.
GATE (Oct 17): GaitBase within 1.0 pp of OpenGait's reported 97.6/94.0/77.4 per condition AND
`evaluate_attack(attack=None)` identical to OpenGait's `evaluate_indoor_dataset`.
GaitSet row (95.8/90.0/75.4) is added when 01b finishes — required before Oct 28.

### 01b_gaitset_training — WS2, due ~Oct 24
Train GaitSet with OpenGait's trainer and `configs/opengait/gaitset_casiab_1gpu.yaml`, 40k
iterations in resumable chunks (save every 2000 iterations, chunks < 3 h). Copy checkpoints to
the team Dataset/Drive after every chunk. Gate checked in Notebook 01.

### 02_pgd_attack — due Oct 20
1. Unit tests: ε=0 reproduces clean; ‖δ‖∞ ≤ ε; x+δ ∈ [0,1]; weights unchanged; seed-stable.
2. Time one PGD step; size the fixed subset so one (attack, ε) sweep finishes in <3 h.
   Record the chosen subset size in configs/attacks.yaml.
3. Pilot 30 probes, then full subset, ε ∈ {2,4,8,16}/255.
4. Save `results/raw/pgd_results.csv`.
Columns: run_id, attack, tier, seed, sample_id, subject, condition, probe_view, n_frames,
eps, steps, clean_rank1, adv_rank1, attack_success, embedding_shift_l2, linf_actual,
l2_actual, n_pixels_perturbed, frac_perturbed, n_frames_perturbed

### 03_edge_masked_pgd — due Oct 22
Hou et al.'s edge mask (distance to boundary < τ=1.0, cv2.distanceTransform); visualise over
10 silhouettes; report mean mask area.
Same PGD, same probes, same ε grid. Save `results/raw/edge_masked_results.csv`.

### 04_temporal_masked_pgd — due Oct 23
Two frame-selection rules: random-k, seeded per sample (mirrors He et al., who sample K
frames at random) and top-k by gradient magnitude.
k ∈ {1,3,5}. Log selected indices. Save `results/raw/temporal_masked_results.csv`.
CHECKPOINT Oct 23: all Tier A CSVs on identical samples.

---

## WS2 — He et al. 2023 reproduction

### 05_wgan_gp_training — due Oct 25
1. Read He et al. sections on G, D, latent prior and loss. Record architecture choices in
   docs/deviations.md.
2. Train WGAN-GP on CASIA-B TRAINING subjects (001–074) silhouettes, 64×44. Never use test
   subjects. Resumable, checkpoint every 30 min to Kaggle Dataset / Drive.
3. Quality checks: sample grids every epoch; visual realism; nearest-neighbour check
   against training silhouettes (no memorisation).
GATE: generated silhouettes look like plausible walking poses. If training is unstable by
Oct 23, reduce resolution or use a simpler DCGAN and record the deviation.

### 06_he_attack — due Oct 28
1. Latent optimisation with MI-FGSM; loss = cosine feature term + D realism term; replace
   K randomly selected frames (seeded). Frame-alteration only.
2. Validate on GaitSet first: compare accuracy drop against He et al.'s reported numbers at
   matching K/N sparsity. Record the gap.
3. Run on GaitBase through `evaluate_attack`, same probes as Tier A.
4. Save `results/raw/he_results.csv` + generated frames for the fingerprint subset.

---

## WS3 — Hou et al. 2025 reproduction & defence

### 07_hou_attack — due Oct 27
1. Read Hou et al.'s method: edge-region selection, spatial/temporal sparsity, optimisation,
   evaluation setup. Fill docs/deviations.md BEFORE coding.
2. Implement, validate against any numbers they report for a comparable setting.
3. Run on GaitBase, same probes, TSR=1.0 (bridge setting) and TSR=0.5. Save
   `results/raw/hou_results.csv`.

### 11_defence_transfer — training Oct 28–Nov 2, evaluation Nov 3–4
1. Read DAGait (Zheng et al., IJCB 2025). Confirm the pre-registered transfer criterion in
   configs/defence.yaml and copy it into the first markdown cell BEFORE any results.
2. Fine-tune GaitBase from its checkpoint with DAGait-style feature-level progressive
   adversarial training (PGD on the embedding, ramped clean/adv mixing). Resumable. Measure time/iteration first and set
   iterations to fit ~20 GPU-hours across teammates' quotas.
3. Clean Rank-1 of defended vs baseline.
4. Evaluate defended model against all five attacks (3 Tier A + 2 Tier B), with ADAPTIVE
   attacks — every attack re-run on the defended model's own gradients.
5. Relate transfer to fingerprint similarity from Notebook 10.
FALLBACK (decide Nov 1): if fine-tuning hasn't converged, stop. Paper drops defence
transfer; title and claims narrow accordingly.

---

## Joint analysis

### 08_attack_comparison — Oct 28–29
Rank-1 vs strength for all five attacks (nominal ε and actual-perturbation axes);
matched-success-rate operating point per attack; breakdown by NM/BG/CL and view;
embedding displacement distributions.

### 09_vulnerability_fingerprints — Oct 29–31
Fingerprint subset: ~60 probes (20/condition) successfully attacked by all five attacks at
the matched operating point. Compute activation shift (hooks), attribution shift
(Grad-CAM + IG), embedding geometry. Save `results/fingerprints/{attack}.parquet`.

### 10_statistics_and_bridge — Nov 1–3
5×5 similarity matrices per fingerprint dimension (Spearman of layer vectors, JS divergence
of strip distributions, CKA profiles, top-k IoU of attribution maps), subject-level
bootstrap CIs, clean-vs-clean and random-mask baselines. Bridge test: He ↔ temporal-sparse
PGD and Hou ↔ edge-masked PGD versus other pairings.
Output `results/tables/fingerprint_similarity.csv`, `results/tables/bridge_test.csv`.

### 12_final_figures — Nov 5–6
Every paper figure and table regenerated from `results/` only.
