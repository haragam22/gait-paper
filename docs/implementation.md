# Implementation plan — two people

Phase-by-phase plan to the CVPR 2027 submission. Derived from `CLAUDE.md`,
`docs/notebook_specs.md` and `docs/execution_plan_v7.docx`. The team is **two people**, so this
file replaces the three-workstream owner column in `notebook_specs.md`. Notebook steps, CSV
columns and hyper-parameters stay in `notebook_specs.md` / `configs/`; this file says **who,
when, what must exist before, and what ends each phase**.

## Hard deadlines (2026, AoE — do not move)

| Date | What |
|---|---|
| **Oct 17** | Gate: clean GaitBase matches OpenGait |
| **Oct 23** | Tier A checkpoint · GAN gate |
| **Oct 28** | Gate: all five attacks on the same subset · He validated on GaitSet |
| **Nov 1** | Gate: defence converging, or drop defence transfer |
| **Nov 4** | All result tables exist |
| **Nov 9** | Internal review |
| **Nov 10** | OpenReview registration (title + abstract) |
| **Nov 15** | Upload (one day of slack before the deadline) |
| **Nov 16** | **Paper deadline** |
| **Nov 23** | **Supplementary deadline** |

Rule: a gate missed by more than one day triggers its fallback (listed in each phase), not a
schedule slip. Downstream dates do not move.

## The split

| | **Person 1 — attack & defend GaitBase** | **Person 2 — He et al. chain & analysis** |
|---|---|---|
| One-line explanation | Everything that perturbs or retrains GaitBase directly: baseline, the three controlled PGD attacks, Hou et al., and the defence | Everything built on top: GaitSet + GAN + He et al. attack, then the comparison, fingerprints and statistics over all five attacks |
| Notebooks | 01, 02, 03, 04, 07, 11 | 01b, 05, 06, 08, 09, 10 |
| Owns in `src/` | `attacks/{pgd,masks,hou_edge}.py`, `defence.py` | `gan/*`, `attacks/he_gan.py`, `fingerprints/*`, `stats.py` |
| GPU-heavy jobs | Tier A sweeps (Oct 20–23), defence fine-tune (~20 GPU-h, Oct 28–Nov 2) | GaitSet training (Oct 17–24) and WGAN-GP (Oct 17–25) at the same time — run one on Kaggle and one on Colab |
| Joint | 12 (figures), paper writing | 12 (figures), paper writing |

GPU heavy work falls in different weeks for each person: P2's training in Oct 17–25, P1's
defence in Oct 28–Nov 2. Free Kaggle + Colab only, each person on their own accounts
(CLAUDE.md compute rules). Check the remaining weekly GPU quota on Kaggle before starting a long
job. Every long job is resumable, so if one person's quota runs out, the other can run the next
chunk from the shared checkpoint on their own account.

### Interfaces — the only cross-person dependencies

| # | What | From → to | Due | Contract |
|---|---|---|---|---|
| I1 | `src/{data,evaluation,utils,checkpointing}.py` | P1 → P2 | Oct 17 | **Frozen after Phase 1.** Changes only via PR reviewed by the other person; `pytest tests` passes |
| I2 | Packed CASIA-B (`CASIA-B-pkl`) | P1 → P2 | Oct 17 | Notebook 01's saved Kaggle output (attach as input) |
| I3 | Fixed attack subset `results/processed/subset_attack.csv` | P1 → P2 | Oct 20 | Written once by Notebook 02, never regenerated. P2 needs it for 06 by Oct 27 |
| I4 | Attack callables + result CSVs: dense, edge, temporal (Oct 23), Hou (Oct 27) | P1 → P2 | Oct 23 / 27 | `attack(model, x, target) -> x_adv` (signature in `src/evaluation.py`); seeded, so P2 regenerates adversarial sequences for fingerprints instead of passing files |
| I5 | He et al. callable + `he_results.csv` | P2 → P1 | Oct 28 | Same signature; P1 needs it for adaptive defence evaluation |
| I6 | `results/tables/fingerprint_similarity.csv` | P2 → P1 | Nov 3 | Output of Notebook 10; used in Notebook 11 step 5 |

Git: each person works on their own branch and merges to `main` by PR. Edit only files you own;
shared files (I1, `configs/attacks.yaml`, `docs/deviations.md`, `paper/`) change in small PRs.
Sync at every gate: a short call to check the gate criteria together.

---

## Phase 0 — Setup ✅ (Oct 15–16)

- Repo, `CLAUDE.md`, configs, docs; OpenGait cloned at the pinned commit `4d228e1`.
- CASIA-B public PNG mirror downloaded; packing verified locally (`docs/reproducibility.md`).
- `src/` foundations + tests (8 pass); Notebooks 01 and 01b written; pushed to GitHub.

---

## Phase 1 — Foundations and clean baseline (Oct 16–17)

**Prerequisites:** Phase 0; GitHub access for both; Kaggle secret `GITHUB_TOKEN` per person;
`trnquanghuyn/casia-b` attached; GPU T4 + Internet on.

| Person 1 | Person 2 |
|---|---|
| Run Notebook 01 on Kaggle (packs CASIA-B, GaitBase_DA-60000, clean eval, OpenGait cross-check). Save Version → share the output (I2). Fill `docs/reproducibility.md` and the findings cell from the gate table. | Read He et al. (G/D, latent prior, loss). Record the G/D architecture in `docs/deviations.md` + `configs/gan.yaml`. As soon as I2 exists: start 01b with a 2,000-iteration timing chunk, then set `CHUNK`. |

**Gate (Oct 17) — stops the line:**
- GaitBase clean Rank-1 within 1.0 pp of 97.6 / 94.0 / 77.4 (NM / BG / CL);
- `evaluate_attack(attack=None)` equals OpenGait's `evaluate_indoor_dataset` (`matches_evaluator` all True);
- Notebook 01 prints `GATE (GaitBase): PASS`.

**If it fails:** P1 debugs (data gaps, fp16, frame limit) before any attack work. P2 is not
blocked: GAN design and GaitSet training don't depend on the GaitBase result.

---

## Phase 2 — Attacks (Oct 18–28)

### Person 1 — controlled attacks + Hou et al.

| Step | NB | Due | Prerequisite | Done when |
|---|---|---|---|---|
| 2.1 PGD | 02 | **Oct 20** | Phase 1 gate | Unit tests pass (ε=0 = clean, ‖δ‖∞ ≤ ε, x+δ ∈ [0,1], weights unchanged, seed-stable); one PGD step timed; subset size in `configs/attacks.yaml`; `subset_attack.csv` written (**I3**); `pgd_results.csv` for ε ∈ {2,4,8,16}/255 |
| 2.2 Edge-masked PGD | 03 | **Oct 22** | 2.1 | Edge mask (τ=1.0, `cv2.distanceTransform`) shown on 10 silhouettes, mean area reported; `edge_masked_results.csv` on the same subset |
| 2.3 Temporal-sparse PGD | 04 | **Oct 23** | 2.1 | random-k and top-k-gradient, k ∈ {1,3,5}, indices logged; `temporal_masked_results.csv` (**I4**, Tier A part) |
| 2.4 Hou et al. | 07 | **Oct 27** | 2.2 (reuses the edge mask) | `docs/deviations.md` filled **before** coding; TSR=1.0 and 0.5 on GaitBase, same subset; gap to Hou's reported trends recorded; `hou_results.csv` (**I4**) |
| 2.5 Defence prep | 11 | Oct 27 | Phase 1 gate | Read DAGait; transfer criterion copied from `configs/defence.yaml` into Notebook 11's first markdown cell **before any result**; `src/defence.py` written; time per iteration measured → number of iterations set to fit ~20 GPU-h |

Tier A runs 2.2 and 2.3 in parallel on Kaggle + Colab if one sweep does not fit a day.

### Person 2 — GaitSet, GAN, He et al.

| Step | NB | Due | Prerequisite | Done when |
|---|---|---|---|---|
| 2.6 GaitSet training | 01b | **Oct 24** | I2 | `GaitSet-40000.pt` in the shared Dataset; P2 runs Notebook 01's GaitSet cell: within 1.0 pp of 95.8 / 90.0 / 75.4 |
| 2.7 WGAN-GP | 05 | **Oct 25** | I2; architecture recorded | Trained on subjects 001–074 only; sample grid per epoch; nearest-neighbour check shows no memorisation |
| 2.8 He et al. attack | 06 | **Oct 28** | 2.6, 2.7, I3 | Validated on GaitSet first (accuracy drop vs. reported numbers at matching K/N, gap recorded); then GaitBase via `evaluate_attack` on the subset; `he_results.csv` (**I5**) |
| 2.9 Fingerprint code | 09 (code only) | Oct 27 | I3, `pgd_results.csv` | `src/fingerprints/*` (hooks incl. pre-hook on `FCs`, Grad-CAM + IG, geometry) tested on the dense-PGD outputs, so Phase 3 is only running it |

**GAN gate (Oct 23):** generated silhouettes look like plausible walking poses.
**Fallback:** reduce resolution or switch to a simpler DCGAN on Oct 23 (not later); record the deviation.

**Phase 2 gate (Oct 28) — stops the line:**
- Five result CSVs (`pgd`, `edge_masked`, `temporal_masked`, `hou`, `he`) on identical `sample_id`s, each
  row with actual L∞/L2, pixels, fraction and frames perturbed.
- GaitSet clean row passes (2.6).
- He et al. validated on GaitSet. **Fallback:** the plan defines none. Decide together on Oct 28: either
  proceed with the measured gap reported in `docs/deviations.md` or mark claim C3 as weakened in
  `paper/claim_evidence_ledger.csv`. Either way the schedule does not move.
- All five attacks importable as callables (I4, I5).

---

## Phase 3 — Comparison, fingerprints, defence training (Oct 28–Nov 2)

**Prerequisites:** Phase 2 gate.

| Person 1 | Person 2 |
|---|---|
| **11 defence training (Oct 28–Nov 2):** fine-tune GaitBase from GaitBase_DA-60000 with embedding-space PGD + ramped clean/adv mixing (DAGait-style); resumable, checkpoint ≤ 30 min, chunks < 3 h. Write the methods for Tier A, Hou and the defence while it trains. | **08 attack comparison (Oct 28–29):** Rank-1 vs strength for all five attacks (nominal-ε and actual-perturbation axes); matched-success-rate operating point per attack; NM/BG/CL and view breakdown; embedding displacement. |
| | **09 fingerprints (Oct 29–31):** fingerprint subset (~60 probes, 20/condition, attacked by all five at the matched point); activation shift, Grad-CAM + IG, embedding geometry; `results/fingerprints/{attack}.parquet`. |

**Phase 3 gate (Nov 1):**
- Fingerprint parquet for all five attacks (P2, due Oct 31).
- Defence (P1): fine-tuning converging → continue. **Fallback:** otherwise stop; the paper drops defence
  transfer (claim C4) and the title and claims narrow. P1 moves to helping with statistics and the paper.

---

## Phase 4 — Statistics, bridge test, defence evaluation (Nov 1–4)

**Prerequisites:** Phase 3 gate.

| Person 1 | Person 2 |
|---|---|
| **11 defence evaluation (Nov 3–4)** (skipped under the fallback): clean Rank-1 defended vs. baseline; all five attacks re-run **adaptively** on the defended model's own gradients; relate transfer to fingerprint similarity (I6); `results/tables/defence_transfer.csv`. | **10 statistics + bridge (Nov 1–3):** 5×5 similarity matrices per fingerprint dimension; subject-level bootstrap (1000 resamples, 95% CI); clean-vs-clean and random-mask baselines; bridge test He ↔ temporal-sparse PGD and Hou ↔ edge-masked PGD vs. other pairings; `fingerprint_similarity.csv` (**I6**, Nov 3), `bridge_test.csv`. |

**Phase 4 gate (Nov 4):** `bridge_test.csv`, `fingerprint_similarity.csv`, and `defence_transfer.csv`
(or the documented fallback) exist; every claim in `paper/claim_evidence_ledger.csv` points to a result
file or is marked dropped.

---

## Phase 5 — Figures and paper (Nov 5–16; supplementary to Nov 23)

**Prerequisites:** Phase 4 gate. The methods sections were drafted during Phases 2–3.

| Date | Person 1 | Person 2 |
|---|---|---|
| Nov 5–6 | 12 final figures, defence + attack-comparison parts (regenerated from `results/` only) | 12 final figures, fingerprint + bridge parts |
| Nov 5–8 | Results: Tier A, Hou, defence transfer; `docs/deviations.md` final | Results: He, fingerprints, bridge test; related work |
| **Nov 9** | Internal review — both read the full draft | |
| **Nov 10** | OpenReview registration: title + abstract (both have OpenReview profiles) | |
| Nov 11–14 | Fix review comments; intro + abstract final | Fix review comments; discussion + limitations |
| **Nov 15** | Upload | |
| **Nov 16** | Paper deadline | |
| Nov 17–23 | Supplementary: per-sample results, extra breakdowns, reproducibility record (`docs/reproducibility.md`) — split by the same ownership | |

**Gate:** every number in the paper traces to a file in `results/` (`claim_evidence_ledger.csv`
status = supported); no number typed from memory (CLAUDE.md rule 8).

---

## Dependency summary

```
Person 1:  01 ─► 02 ─► 03 ─► 04 ─► 07 (Hou) ─► 11 train ─────────► 11 eval ─► 12 + paper
            │     │                │              ▲                    ▲
            I2    I3               I4             I5                   I6
            ▼     ▼                ▼              │                    │
Person 2:  01b + 05 (GAN) ─► 06 (He) ─────────────┴─► 08 ─► 09 ─► 10 ──┴────► 12 + paper
```

P2 waits on P1 only for I2 (Oct 17), I3 (Oct 20) and I4 (Oct 23 / 27). P1 waits on P2 only for I5
(He callable, Oct 28) and I6 (fingerprint similarity, Nov 3). Before Oct 28 each person can keep
working whenever the other is late, except for 06 needing the subset file.
