# Paper outline (CVPR 2027, 8 pages + refs)

Working title: Cross-Attack Mechanistic Analysis of Adversarial Vulnerabilities in Gait Recognition
(add "and Defence Transfer" only if C4 is supported)

1. Introduction — gait models are attacked and defended by many papers, each with its own
   attack family and setup; no defence shown to transfer; nobody has asked whether the
   attacks exploit the same internal representations. Contributions = only what C1–C4 support.
2. Related work — gait attacks (He 2023, Hou 2025, Ali 2025, Saoudi 2026); gait defences
   (GaitReload 2023, Ji 2024, DAGait 2025); gait interpretability (DisRec 2026);
   adversarial attribution in other domains; attack-specific signatures (Souri et al. 2021,
   arXiv 2110.06802) — closest prior work on telling attacks apart, position against it.
3. Method — common setup; Tier A masked-PGD family; Tier B reproductions (He 2023, Hou 2025); activation + attribution fingerprints; bridge test.
4. Experimental setup — CASIA-B protocol, GaitBase, budgets, metrics, bootstrap design.
5. Results — attack comparison; fingerprint similarity.
6. Defence transfer.
7. Discussion & limitations — attribution ≠ causation; one model, one dataset; Tier B fidelity limits;
   Tier A are controlled variants, not reproductions.
8. Conclusion.
