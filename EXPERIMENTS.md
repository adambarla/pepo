# PEPO experiment tracker

Last updated: 2026-09-15

This tracks only runs still needed and observations from completed runs. For
each new run, record the exact command/config, seed, Clariden job ID, W&B run,
and the main result in the row below.

## Runs still needed

| ID | Run | Status | What to record / decide |
|---|---|---|---|
| R-001 | Margin-only shifted-sigmoid DPO on Mistral-7B, matching the Tulu protocol | recommended | Fill the mechanism-isolation gap; compare the full epoch curve, not only the best epoch |
| R-002 | Repeat the Tulu `L=4` overlap-subset experiment, or run it on a second backbone | recommended | Save the exact split recipe, seed, union coverage, pairwise overlap, and E16 result |
| R-003 | Recover or recompute the missing Tulu margin-only E14 value | needed for the table | Do not interpolate; record the source run or rerun it |
| R-004 | True bootstrap control (`replace=True`) | optional | Only run if we want to distinguish bootstrap from the current `overlap_subsample` mode |

No new run is currently needed for final-epoch evaluation, MT-Bench transfer,
early stopping, the controlled `pi_data`/`pi_ref` experiments, inference
latency, or the controlled-bandit `L` experiment.

## Observations

- **Margin-only, Tulu:** at `L=1, alpha=0.1`, shifted-sigmoid DPO is 69.9% at
  E8 and 64.8% at E16; standard DPO is 68.7% at E8 and 60.8% at E16. PEPO
  with `L=3, alpha=0.1` is 75.0% at E8 and 73.8% at E16. The margin helps,
  but does not explain PEPO's late-training stability by itself.
- **Overlap subset, Tulu:** the documented `L=4` run has approximately 25%
  pairwise overlap and 68.4% union coverage. At E16 it reaches 75.9%, versus
  73.4% for disjoint PEPO. This is empirical evidence only; the current
  independence-based proof does not automatically apply to overlapping members.
- **Fixed endpoint:** at the final epoch, PEPO is best on all four evaluated
  backbones in both AlpacaEval2 and MT-Bench. Best-epoch rankings are mixed.
- **MT-Bench:** the late-training pattern transfers beyond AlpacaEval2, but the
  paper should retain the mixed best-epoch cases rather than claim universal
  oracle-checkpoint superiority.

## Run log

| ID | Implementation commit | Command/config | Seed | Job ID | W&B URL | Result |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |
