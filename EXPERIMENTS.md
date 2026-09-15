# PEPO experiment registry

Last updated: 2026-09-15

This is the lightweight progress tracker for the PEPO implementation and
paper. It records the scientific question, the evidence already available,
where that evidence is used, and the next action. Raw checkpoints, generated
responses, W&B histories, and Slurm logs stay outside Git.

The initial objective is to turn the rebuttal work into a defensible ICLR 2027
evidence plan: preserve the results that answer reviewer concerns, identify
only the experiments that close real evidence gaps, and keep every new run
traceable to its code, configuration, data, and paper use. This is deliberately
a lightweight registry inside the existing implementation repository; it is
not a second `lowrank-application` repository or a reason to move generated
artifacts into Git.

## Status key

- `complete`: result exists and is usable, subject to the canonical-number
  check below.
- `partial`: some results exist, but coverage or metadata is incomplete.
- `running`: jobs or analysis are active.
- `pending`: useful follow-up, not required for the current paper.
- `blocked`: cannot proceed until an implementation or metadata issue is fixed.

## Current repository snapshot

- Local implementation repository: branch `feat/visualisations`; visualization
  files and a few analysis helpers are currently modified/untracked.
- Clariden checkout: `/users/barla/projects/pepo`, branch
  `feat/bootstrap-split`; bootstrap/split changes and several launch scripts
  are currently uncommitted.
- W&B project used by the implementation: `pepo`.
- Hydra preserves resolved configs and overrides under each run's
  `outputs/hydra/.../.hydra/` directory.
- Remote split tests currently pass: 11 tests.

The paper and implementation are separate version-control units. Link them
using the experiment ID, implementation commit, W&B run, and paper figure or
table rather than moving the repositories.

## Experiment registry

| ID | Question | Status | Existing evidence | Paper use | Next action |
|---|---|---|---|---|---|
| E-001 | Does PEPO avoid late-training over-optimization across backbones and ensemble sizes? | complete | `figures/main_figure.pdf`, `figures/alpaca_win_rate_L_ablation.pdf`, `figures/mtbench_win_rate_L_ablation.pdf`; visualization notebook | Main learning curves and ensemble-size discussion | Freeze the canonical curves and add the step-axis version to the appendix |
| E-002 | Is the shifted-sigmoid loss alone sufficient without an ensemble? | partial: Tulu result complete | Tulu margin-only result and full epoch values in the separate `hackmd` reviewer notes | Main mechanism-isolation ablation | Fill the missing E14 value, then preferably repeat on Mistral-7B |
| E-003 | Is disjoint partitioning necessary in practice? | partial | Tulu `L=4` disjoint/overlap values are recorded in `../hackmd/reviewer_MPAm.md`; overlap reaches 75.9% at E16 versus 73.4% disjoint | Main-text summary plus full appendix table/curve | Record the exact split recipe and repeat once or evaluate a second backbone |
| E-004 | Are the gains robust at a fixed, non-oracle endpoint? | complete: canonicalization pending | `figures/table_alpaca_final_epoch.tex` and `figures/table_mtbench_final_epoch.tex`; final PEPO result is best on all four models in both tables | Primary main-text comparison | Make final epoch primary; move best-epoch tables to the appendix |
| E-005 | Does the late-training pattern transfer beyond AlpacaEval2? | complete | MT-Bench best/final tables and `figures/mtbench_win_rate_methods.pdf` | One main-text sentence and full appendix table | Verify judge/config metadata and retain the mixed best-epoch cases honestly |
| E-006 | Can validation-loss early stopping replace PEPO? | complete | `fig:early_stop` in the current ICLR draft and the corresponding appendix discussion | Appendix control experiment | Keep as an appendix control; do not spend new compute here |
| E-007 | Does PEPO tolerate unknown mismatch between `pi_data` and `pi_ref`? | complete | Controlled-setting figures under `figures/perfspidataneqpiref.pdf`, `figures/perfspidataeqpiref.pdf`, and related greedy plots | Appendix controlled experiment | Check captions and distinguish known from unknown `pi_data` clearly |
| E-008 | What is the inference-quality/latency tradeoff? | complete | Current ICLR generation-time plots and the rejection-sampling cells in `notebooks/visualization.ipynb` | Appendix only | Keep out of the central ablation story |
| E-009 | How does the theoretical L tradeoff appear in a controlled bandit? | complete | `figures/PEPOabl.pdf` and the controlled-setting notebook | Appendix support for L-selection guidance | No new run required |
| E-010 | Can the implementation run overlapping ensemble members without overwriting checkpoints? | running/blocked on naming semantics | Clariden `feat/bootstrap-split`; split-mode config, checkpoint suffix, launch scripts, and split tests | Reproducibility infrastructure, not a paper result by itself | Decide whether the mode is `overlap_subsample` or true `bootstrap`; commit the implementation before production runs |

## Results currently safe to state

### Margin-only Tulu result

At `L=1, alpha=0.1`, shifted-sigmoid DPO reaches 69.9% at E8 and ends at
64.8% at E16. Standard DPO reaches 68.7% and ends at 60.8%. PEPO with
`L=3, alpha=0.1` reaches 75.0% and ends at 73.8%.

Interpretation: the margin helps, but does not explain PEPO's late-training
stability by itself.

### Overlap result

The documented Tulu `L=4` overlap run uses approximately 25% pairwise overlap
and covers 68.4% of the training set in its union. Its E16 win rate is 75.9%,
compared with 73.4% for disjoint PEPO.

Interpretation: disjointness is required by the current independence-based
proof, but need not be required for empirical robustness. This is not evidence
that the existing theorem applies to overlapping members.

### Fixed-endpoint results

The final-epoch tables report PEPO as the best method on all four backbones in
both AlpacaEval2 and MT-Bench. Best-epoch rankings are mixed, so the paper
should describe PEPO as more robust at a fixed late-training endpoint rather
than universally best at every oracle-selected checkpoint.

## Implementation follow-ups

1. Freeze one canonical result sheet. Resolve the Llama/Tulu discrepancy
   between the active ICLR table (76.1) and the newer table fragments (75.0),
   and standardize model names, judge, epoch, and decoding configuration.
2. Integrate the final-epoch and margin-only results into the active ICLR
   source. The table fragments under `implementation/figures/` are not a
   substitute for wiring them into `overleaf/iclr2027/`.
3. Correct the split terminology. The current Clariden code uses
   `rng.choice(..., size=N/L, replace=False)`, which is overlapping
   subsampling, not standard bootstrap sampling with replacement.
4. Keep `scripts/slurm/` and the newer top-level `slurm/` from becoming two
   competing launch trees. Choose one canonical location after the current
   jobs are safely checkpointed.
5. Add the experiment ID to W&B tags, Slurm job names, and the run notes. The
   existing Hydra `.hydra/config.yaml` and `.hydra/overrides.yaml` files are
   already sufficient as the per-run configuration record.

## Recommended next experiments

- Required for a clean paper: canonicalize and integrate existing results.
- Strongly recommended: run margin-only on Mistral-7B using the same protocol.
- Useful robustness check: repeat the overlap split once or run it on a second
  backbone.
- Optional: implement a true bootstrap mode (`replace=True`, usually with a
  clearly specified per-member sample size). Do not call the current
  `N/L, replace=False` mode bootstrap.

## Experiment entry template

Copy this template into a new entry or append it to the registry before
launching a new study:

```text
ID / name:
Question:
Hypothesis:
Status:
Implementation commit:
Branch:
Train command:
Eval command:
Hydra overrides:
Seed / dataset revision / model revision:
Slurm job IDs:
W&B run URLs:
Checkpoint or Hub names:
Output root:
Primary result:
Paper figure/table:
Caveats:
Next action:
```

Fields for old experiments that are not present in the repository should be
filled from W&B or Clariden logs when the result is regenerated; do not invent
run IDs or checkpoint paths.
