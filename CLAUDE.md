# CLAUDE.md — instructions for Claude Code on this repo

Read this file fully before writing any code. It is the contract for how this project is built.

## The project in one paragraph

Cross-Attack Mechanistic Analysis of Adversarial Vulnerabilities in Gait Recognition.
We attack one gait model (GaitBase, from OpenGait) on one dataset (CASIA-B) with
(a) controlled attack variants that differ ONLY in where the perturbation is allowed, and
(b) reproductions of two published gait attacks. We measure how each attack changes the
model's internal activations and attributions ("vulnerability fingerprints"), test whether
each published attack's fingerprint matches its controlled counterpart, and test whether
fingerprint similarity explains which defences transfer across attacks.
Target venue: CVPR 2027 only — registration Nov 10, paper Nov 16, supplementary Nov 23
(2026, AoE). Do not prepare a parallel submission to any other venue.

## Research question (do not drift from this)

Do structurally different adversarial attacks on a gait recognition model disrupt the same
internal representations, and does the similarity of their vulnerability fingerprints
explain cross-attack defence transfer?

## Scope

- Model: GaitBase (OpenGait). GaitSet ALSO used, only to validate our reproduction of
  He et al. against the numbers they report (they attacked GaitSet). OpenGait releases a
  CASIA-B GaitBase checkpoint (GaitBase_DA-60000.pt) but NO CASIA-B GaitSet checkpoint, so
  GaitSet is trained with OpenGait's own trainer (Notebook 01b, configs/opengait/).
- Dataset: CASIA-B, OpenGait pretreated silhouettes, official 74/50 split, standard
  probe/gallery protocol (gallery nm-01..04; probes nm-05/06, bg-01/02, cl-01/02;
  identical-view gallery entries excluded).
- Attacks — two tiers (see "Attack design"):
  - Tier A, controlled: dense PGD, edge-masked PGD, temporal-sparse PGD.
  - Tier B, published: He et al. 2023 (GAN latent attack), Hou et al. 2025 (edge-oriented).
- Fingerprints: activation shift (forward hooks) + attribution shift (Grad-CAM, IG).
- Defence: one embedding-space adversarial-training defence (DAGait-style: PGD on the
  EMBEDDING with progressive clean/adv mixing, not input-space PGD), fine-tuned from the
  GaitBase checkpoint, plus the undefended baseline.
- Out of scope unless core work finishes early: DeepGaitV2, other datasets, extra defences.

## Non-negotiable engineering rules

1. Reuse OpenGait's model code, checkpoints and evaluation protocol. Never reimplement
   GaitBase, GaitSet or the CASIA-B split. Clean numbers must match OpenGait's published ones.
2. Notebooks are for running and plotting. All logic lives in `src/`. A function is never
   copy-pasted between notebooks.
3. Every attack — Tier A and Tier B — goes through ONE shared function
   `evaluate_attack(...)` in `src/evaluation.py`. Clean evaluation is `attack=None`.
4. Every run saves config, seed, OpenGait commit, checkpoint id, and per-sample rows (not
   just aggregates) to `results/raw/`. CSVs are appended, never overwritten.
5. Model parameters are frozen during attacks (`requires_grad_(False)`, `eval()`). Assert
   weights are unchanged after every attack run.
6. Zero-perturbation must reproduce clean results exactly. Unit test.
7. Fixed, seeded sample subsets (stratified by NM/BG/CL) are shared across ALL attacks so
   every comparison and fingerprint uses identical samples.
8. Do not invent numbers, results or citations anywhere — code, comments, docs, paper.

## Compute rules (free Kaggle + free Colab only)

- One Kaggle account and one Colab account per teammate. Never multiple accounts per person.
- CASIA-B silhouettes and checkpoints live in a private Kaggle Dataset shared within the
  team, and mirrored on Google Drive for Colab.
- Every long job (GAN training, adversarial training, full attack sweeps) must be
  resumable: checkpoint at least every 30 minutes, resume automatically from the latest
  checkpoint, and write per-sample result rows incrementally so a disconnect loses at most
  one batch.
- Sessions are capped (Kaggle ~12 h per session; Colab free can disconnect any time).
  Design every job to finish in chunks under 3 hours.
- Attack sweeps run on the fixed subset, not all ~3,300 probes. Time one PGD step in
  Notebook 02 and size the subset from that measurement.
- Use mixed precision only for forward passes in evaluation; keep attack gradients in fp32.

## Threat model and attack objective (Tier A — implement exactly this)

- Input: silhouettes as float in [0,1], shape [B, T, H, W] (64x44 after OpenGait
  pretreatment). Perturbed inputs clamped to [0,1].
- Recognition: nearest gallery sequence by part-averaged Euclidean distance on the
  GaitBase embedding [N, C, P].
- Untargeted loss (maximise): margin = d(probe, nearest gallery sample of TRUE identity)
  − d(probe, nearest gallery sample of any OTHER identity).
  Success = clean top-1 correct AND adversarial top-1 wrong.
  "Gallery" for the loss and for success = the POOLED cross-view gallery (all nm-01..04
  sequences whose view != probe view). Reported Rank-1 uses OpenGait's protocol exactly
  (per gallery view, identical view excluded, averaged): `rank1_viewavg` columns.
- Budget: L∞, ε ∈ {2, 4, 8, 16}/255; 50 steps; step = ε/10; one random start.

## Attack design

### Tier A — controlled (one optimiser, three support masks)

| Family | Support mask | Mirrors |
|---|---|---|
| Dense PGD | all pixels, all frames | Madry et al. 2018 |
| Edge-masked PGD | Hou et al.'s exact edge region (distance to boundary < τ=1.0, cv2.distanceTransform), all frames | Hou et al. 2025 |
| Temporal-sparse PGD | all pixels in k frames only; frames chosen to mirror He et al. (k frames at random, seeded) AND, separately, top-k by gradient magnitude | He et al. 2023 |

Tier A varies ONLY the support, so differences between Tier A fingerprints are caused by
where the perturbation is allowed.

### Tier B — published attacks (reproduced as faithfully as the papers allow)

He et al., Pattern Recognition 2023 (arXiv 2002.09674), no public code:
- Train a WGAN-GP silhouette generator on CASIA-B TRAINING subjects only (001–074).
- Optimise only the latent vector Z (sampled on a sphere) with MI-FGSM.
- Loss = cosine-similarity term on gait features + discriminator BCE realism term.
- Replace K randomly selected frames (the paper samples the K indices at random, not the
  first K): X* = M·G(Z) + (1−M)·X. Frame-alteration mode only.
- Paper values: V=128, 100 MI-FGSM iterations, λ=1000, μ=1.0, α=0.1, WGAN-GP 16k iterations.
  G/D architecture is not given → ours, recorded in docs/deviations.md (configs/gan.yaml).
- Validation: run it against GaitSet and compare the accuracy drop to their reported
  numbers before running it against GaitBase.

Hou et al., IJCV 2025, no public code:
- Implement from the paper's method section (read it first; fill docs/deviations.md).
  Edge region τ=1.0; edge pixels replaced by sigmoid(z); Adam lr 1.0, 20 iterations;
  m=10, λ=20 (configs/attacks.yaml `hou`). Hou is ALSO temporally sparse (TSR): run TSR=1.0
  as the bridge-test setting (isolates edge support) and TSR=0.5 as a secondary row.
- Validation: their GaitBase is a reduced variant they trained themselves (2 residual
  blocks, fewer channels), not OpenGait's checkpoint, so match trends, report the gap.

Never call a Tier A variant by a published attack's name. Every deviation from a Tier B
paper goes into docs/deviations.md.

Always log per attack: nominal ε (Tier A), actual L∞, actual L2, pixels perturbed,
fraction of input perturbed, frames perturbed. Equal ε ≠ equal strength.

## Fingerprints

Computed on the fixed fingerprint subset: probes successfully attacked by every attack at
matched success rate.

(GaitBase's HorizontalPoolingPyramid is not an nn.Module: capture its output as the input
of `FCs` with a forward pre-hook.)

1. Activation shift (primary, cheap): forward hooks on every conv block and on the
   horizontal-pooling output. Per layer and per horizontal strip: normalised L2 distance
   ‖a_adv − a_clean‖ / ‖a_clean‖, and linear CKA between clean and attacked activations.
2. Attribution shift: Captum layer-wise Grad-CAM + Integrated Gradients. Scalar target
   s(x) = −d(embedding(x), mean gallery embedding of the probe's TRUE identity, identical
   view excluded). Same target for clean and every attack.
3. Embedding geometry: cosine shift, distance to true-identity centroid, distance to nearest
   wrong centroid.

Save per-sample fingerprints, never only aggregates. Attribution and activation change are
evidence of changed sensitivity/representation, NOT proof of a causal circuit.

## Bridge test (the key contribution)

Does each Tier B attack's fingerprint match its Tier A counterpart (He ↔ temporal-sparse
PGD, Hou ↔ edge-masked PGD) more than it matches the other Tier A variants? Report with
subject-level bootstrap CIs.

## Statistics

Subject-level bootstrap (resample subjects, not sequences), 1000 resamples, 95% CIs.
Baselines: clean-vs-clean variability (two clean sequences of the same subject) and a
random-mask null of equal perturbed area.

## Repo layout

```
configs/      baseline.yaml attacks.yaml gan.yaml attribution.yaml defence.yaml
notebooks/    01..12 (see docs/notebook_specs.md)
src/          data.py evaluation.py utils.py checkpointing.py
              attacks/{pgd.py,masks.py,he_gan.py,hou_edge.py}
              gan/{wgan_gp.py,train.py}
              fingerprints/{activations.py,attribution.py,geometry.py}
              stats.py defence.py
results/      raw/ processed/ fingerprints/ tables/ figures/
paper/        outline.md claim_evidence_ledger.csv
docs/         notebook_specs.md deviations.md reproducibility.md execution_plan_v7.docx
```

## Build order

Three workstreams run in parallel (see docs/notebook_specs.md). Within a workstream, do
not start a step until the previous step's gate passes. Shared foundations first:
`src/utils.py`, `src/checkpointing.py`, `src/data.py`, `src/evaluation.py` and
Notebook 01 — every workstream depends on them.

## Style

Python 3.10+, PyTorch, type hints, small functions, docstrings stating tensor shapes.
pandas for result tables. No hidden global state. Every notebook's first cell records date,
seed, versions, GPU, platform (Kaggle/Colab), OpenGait commit.
