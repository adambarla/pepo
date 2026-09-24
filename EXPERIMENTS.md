# PEPO experiment tracker

Last updated: 2026-09-21

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
| R-005 | Quick response-level rejection-sampling diagnostic on existing L=4 checkpoints | diagnostic complete | Mistral and Tulu AlpacaEval2 checks plus a matched Tulu MT-Bench check completed. The results support a cost-based motivation for token-level PEPO, not a claim that response-level sampling is uniformly worse |
| R-006 | Response-level acceptance and token-level generation mechanics on Tulu `L=3` checkpoints | diagnostic complete | Strict-minimum acceptance collapses at epoch 3; token-level generation avoids the rejection loop; the practical mean-minus-standard-deviation rule clips most raw ratios at 1 and is not an exact sampler |
| R-007 | Theory-aligned response-level mean-minus-standard-deviation diagnostic | smoke complete | The exact probability-space `eta=sqrt(3)` path hit the four-trial cap on all four smoke prompts; this is a diagnostic, not a quality result |
| R-008 | Re-measure generation latency with exact response-token counts | completed | Matched Tulu E4/E5 token-level and practical response-level runs completed on four GH200s; interpret both wall time per prompt and time per final response token |
| R-009 | Matched L=4 AlpacaEval2 sweep for epochs 1 to 6 and three samplers | in progress | Finish strict response-level generation, then score strict, practical mean-minus-standard-deviation, and token-level outputs against one shared epoch-0 reference set; record win rate, valid comparisons, wall time, final-response tokens, and milliseconds per final-response token |

No new run is currently needed for final-epoch evaluation, MT-Bench transfer,
early stopping, the controlled `pi_data`/`pi_ref` experiments, inference
or the controlled-bandit `L` experiment. Full E4/E5 AlpacaEval2 and MT-Bench
records already exist in the W&B cache, so do not rerun them only to extend a
table.

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
- **Response-level diagnostic, Mistral:** at epoch 1 with `L=4`,
  mean-minus-standard-deviation sampling (`eta=0.1`, four trials) accepted on
  average with probability `0.982` and used `1.0` trial per prompt. On only
  eight AlpacaEval prompts, the local Llama-3-70B judge gave `0/8` wins and a
  `6.44%` length-controlled win rate. This was evaluated against AlpacaEval's
  default GPT-4 reference, not the paper's initial-SFT reference, so it is a
  smoke test rather than a paper result.
- **Response-level diagnostic, Tulu AlpacaEval2:** with the same practical rule
  at epoch 1 (`L=4`, `eta=0.1`, four trials, 16 prompts), the raw win rate was
  `18.75%` and the length-controlled win rate was `39.36%`. AlpacaEval warned
  about the large raw/length-controlled discrepancy. This run also used the
  default GPT-4 reference and is not a replacement for the main metric.
- **Matched Tulu MT-Bench diagnostic:** on four two-turn questions at epoch 1,
  response-level rejection sampling scored `6.75/10` (turns `8.0/5.5`) and
  token-level PEPO scored `6.25/10` (turns `8.0/4.5`) under the same local
  Llama-3-70B judge. Two response-level prompts were truncated to the 512-token
  input limit. The tiny matched comparison shows similar quality, with no basis
  for claiming a response-level quality deficit.
- **Diagnostic conclusion:** the new runs do not justify saying that
  response-level rejection sampling produces poor answers. They support the
  narrower claim that token-level PEPO is the practical default because it
  avoids response-level rejection sampling's high worst-case inference cost;
  the existing full-scale latency measurements remain the evidence for that
  claim.
- **Strict response-level mechanics:** with Tulu `L=3`, strict-minimum
  acceptance fell from `avg_alpha=0.213` and `4.5` actual trials at E1 to
  `0.0232` and `37.1` trials at E3 under `max_trials=64`; `28.1%` of E3
  prompts hit the cap. A matched E3 run with `max_trials=256` had `0/8` cap
  hits and `39.4` actual trials on average, confirming that the 16-trial cap
  censors a genuine low-acceptance process. Duplicate proposals were `0%`
  at E1, `0.064%` at E3 with 64 trials, and `0%` in the 256-trial run.
- **Token-level mechanics:** E1 and E3 generation used one direct greedy
  decoding pass over the ensemble, with no rejection trials. The E3 output
  was saved successfully before Slurm reported a GH200 bus error during
  teardown, so it is usable as a generation diagnostic but not as a clean
  runtime benchmark.
- **Practical mean-minus-standard-deviation rule:** at E3, `96.3%` of raw
  ratios exceeded `1` and some overflowed; a matched finite-log run showed
  `100%` above `1`. At E1, `93.8%` exceeded `1`. The implementation therefore
  relies heavily on `alpha` clipping and should be presented only as a
  heuristic latency variant, not as exact rejection sampling.
- **Exact probability-space smoke test:** for Tulu E4 with `L=4`,
  `eta=sqrt(3)`, and four trials, all four prompts hit the trial cap and none
  were accepted by rejection. This is too small to estimate quality, but it
  shows that the theory-aligned sampler is not represented by the cheap
  practical `eta=.1` run.
- **Matched latency:** on E4, token-level PEPO took `1009.018 s` for 128
  prompts and the practical response-level rule took `680.878 s`. On E5,
  token-level took `740.040 s` and response-level with `max_trials=64`
  took `795.820 s`. The E5 response-level run was slower per prompt
  (`6.22 s` versus `5.78 s`) but produced longer responses, so its
  final-token-normalized cost was slightly lower (`37.846` versus
  `40.015 ms/token`).
- **Shared-backbone explanation:** both generators use one shared backbone
  with four adapters per GPU, replicated over four GPUs. Token-level PEPO
  evaluates all four adapters autoregressively at every token. Response-level
  sampling proposes with adapter 0 and scores completed candidates with all
  four adapters in batched teacher-forced passes.
- **E5 practical retries:** with `max_trials=64`, response-level sampling
  averaged `1.07` proposals per prompt, with `0%` cap hits and
  `0%` duplicate proposals. The practical rule therefore did not exhibit a
  large retry blow-up at E5, even though one worker became the wall-time
  bottleneck.

- **Theory-aligned eta:** the formal probability-space mean-minus-standard-
  deviation construction uses `eta=sqrt(L-1)`, which is `sqrt(3)` for `L=4`.
  The default practical implementation instead uses a log-probability
  approximation. An opt-in exact probability-space path has been added for
  the diagnostic; do not describe `eta=sqrt(3)` in the default path as a
  formal guarantee.
- **R-009 sweep:** all runs use the first 128 AlpacaEval2 prompts, seed 42,
  L=4, the Tulu-3-8B SFT checkpoints, four GH200 GPUs, and a 1024-token
  response limit. Strict and practical mean-minus-standard-deviation
  sampling use 128 maximum proposals, temperature 1, and top-p 0.9. The
  token-level generator uses its default greedy decoding. The verified
  generation cost so far is:

  | sampler | E1 | E2 | E3 | E4 | E5 | E6 |
  |---|---:|---:|---:|---:|---:|---:|
  | strict response-level, ms/final token | 107.553 | 263.521 | 1042.778 | pending | pending | pending |
  | practical mean-minus-standard-deviation, ms/final token | 21.419 | 20.356 | 26.480 | 29.825 | 26.689 | 33.570 |
  | token-level, ms/final token | 44.476 | 46.356 | 54.257 | 56.616 | 40.015 | 30.783 |

  The earlier sampler-specific win rates are not comparable because each
  evaluation used a different epoch-0 reference response file. Common-
  reference rescoring is running in jobs 3473435 (mean-minus-standard-
  deviation), 3473436 (token-level), and 3473437 (strict, after the strict
  merges). The generation jobs are 3459497, 3460658, and 3460655 for strict
  E1 to E3; 3461392 and 3461402 for practical mean-minus-standard-deviation;
  and 3461403, 3458049, and 3458654 for token-level generation. Strict E4
  to E6 are split into chunk jobs and remain in progress.

## Run log

| ID | Implementation commit | Command/config | Seed | Job ID | W&B URL | Result |
|---|---|---|---|---|---|---|
| R-005-gen | working tree, no commit | `eval.slurm`: `mistral7b`, `deppo`, `L=4`, `a=b=0.1`, `e=1`, `ns=8`, `mt=256`, `BestOfN`, `mean_std`, `eta=0.1`, `max_trials=4`, `top_p=0.9`, `stop_after_generation=true` | 42 | 3451590 | [8mbk78ff](https://wandb.ai/pepo-team/pepo/runs/8mbk78ff) | Completed in 5:56; `avg_alpha=0.982`, `avg_try=1.0`; responses saved |
| R-005-score | working tree, no commit | Same generated responses as R-005-gen; `stop_after_generation=false`, local `llama3_70b_judge`, `ns=8` | 42 | 3451634 |  | Completed in 8:24; win rate `0.0`, length-controlled win rate `0.0644`, `n=8` |
| R-005-tulu-alpaca | working tree, no commit | `eval.slurm`: `llama8b`/Tulu, `deppo`, `L=4`, `a=b=0.1`, `e=1`, `ns=16`, `mt=1024`, `BestOfN`, `mean_std`, `eta=0.1`, `max_trials=4`, `top_p=0.9`, local `llama3_70b_judge` | 42 | 3451733 |  | Completed; raw win rate `18.75%`, length-controlled win rate `39.36%`, `n=16`; default GPT-4 AlpacaEval reference; judge warned about the discrepancy |
| R-005-tulu-mt-response | working tree, no commit | `eval.slurm`: Tulu, `deppo`, `L=4`, `a=b=0.1`, `e=1`, `ns=4`, `mt=512`, `BestOfN`, `mean_std`, `eta=0.1`, `max_trials=4`, `top_p=0.9`, MT-Bench single-score judge | 42 | 3451778 |  | Completed; score `6.75/10`, turn scores `8.0/5.5`, `n=4` questions and 8 judged turns |
| R-005-tulu-mt-token | working tree, no commit | Same Tulu MT-Bench setup as R-005-tulu-mt-response, with `deppo`'s default greedy token-level generator | 42 | 3451814 |  | Completed; score `6.25/10`, turn scores `8.0/4.5`, `n=4` questions and 8 judged turns |
| R-005-tulu-mt-wrong-model | working tree, no commit | Mistaken control using `model=reppo`; failed during initialization before generation | 42 | 3451801 |  | Failed with an unsupported `train_batch_size` argument; no result |
| R-006-strict-e1-e3 | working tree, no commit; diagnostic counters added | Tulu `L=3`, `a=b=0.1`, disjoint, AlpacaEval generation only, `ns=32`, `mt=512`, `top_p=1.0`, strict `BestOfN`, `max_trials=64`, E1 and E3 | 42 | 3452056, 3452102 | [p2bz2k75](https://wandb.ai/pepo-team/pepo/runs/p2bz2k75), [dpvlxj21](https://wandb.ai/pepo-team/pepo/runs/dpvlxj21) | E1 `avg_alpha=0.2129`, `avg_try=4.5`, no caps; E3 `avg_alpha=0.0232`, `avg_try=37.1`, `28.1%` caps; duplicate rates `0%` and `0.064%` |
| R-006-strict-e3-256 | working tree, no commit; diagnostic counters added | Same E3 setup with `ns=8`, `max_trials=256` | 42 | 3452236 | [oowk0r4s](https://wandb.ai/pepo-team/pepo/runs/oowk0r4s) | `avg_alpha=0.0246`, `avg_try=39.4`, `0%` caps, `0%` duplicate proposals; all 8 prompts accepted by rejection |
| R-006-token-e1-e3 | working tree, no commit | Tulu `L=3`, same prompts and `mt=512`, default greedy token-level generator, E1 and E3 | 42 | 3452159, 3452168 | [5o6a69en](https://wandb.ai/pepo-team/pepo/runs/5o6a69en), [ncmzyif8](https://wandb.ai/pepo-team/pepo/runs/ncmzyif8) | E1 generated in `11:28`; E3 responses saved after about `7:10` generation, but Slurm later reported a GH200 teardown bus error |
| R-006-meanstd-raw | working tree, no commit; raw-ratio and log-probability counters added | Tulu `L=3`, E1/E3, `ns=4`, `mt=64`, `top_p=1.0`, `BestOfN`, `mean_std`, `eta=0.1`, `max_trials=4` | 42 | 3452296, 3452283 | [rhejn6r2](https://wandb.ai/pepo-team/pepo/runs/rhejn6r2), [uzkydqn7](https://wandb.ai/pepo-team/pepo/runs/uzkydqn7) | E1 raw `alpha>1` rate `93.8%`; E3 `100%` in the finite-log check. No non-finite proposal or target log-probabilities; clipping is the dominant behavior |
| R-007-exact-e4 | working tree, no commit; exact probability-space path and timing instrumentation added | Tulu `L=4`, E4, `ns=4`, `mt=64`, `eta=sqrt(3)`, `probability_space_mean_std=true`, `max_trials=4`, `top_p=.9` | 42 | 3458565 |  | 38.266 s / 240 final response tokens = 159.440 ms/token; all four prompts hit the trial cap |
| R-008-token-e4 | working tree, no commit; timing instrumentation added | Tulu `L=4`, E4, `ns=128`, `mt=1024`, token-level, four GH200s, `stop_after_generation=true` | 42 | 3458049 |  | 1009.018 s / 17,822 final response tokens = 56.616 ms/token; output saved before Slurm teardown exit 7 |
| R-008-response-e4 | working tree, no commit; timing instrumentation added | Same E4 setup, practical `mean_std`, `eta=.1`, `max_trials=16`, `temperature=1`, `top_p=.9` | 42 | 3458050 |  | 680.878 s / 20,164 final response tokens = 33.767 ms/token; 1.18 proposals/prompt, 0% caps, 0% duplicates |
| R-008-token-e5 | working tree, no commit; timing instrumentation added | Tulu `L=4`, E5, `ns=128`, `mt=1024`, token-level, four GH200s, `stop_after_generation=true` | 42 | 3458654 |  | 740.040 s / 18,494 final response tokens = 40.015 ms/token; output saved before Slurm teardown exit 7 |
| R-008-response-e5 | working tree, no commit; timing instrumentation added | Same E5 setup, `backbone.generator_batch_size=32`, practical `mean_std`, `eta=.1`, `max_trials=64`, `temperature=1`, `top_p=.9` | 42 | 3458835 |  | 795.820 s / 21,028 final response tokens = 37.846 ms/token; 1.07 proposals/prompt, 0% caps, 0% duplicates |

## Latency measurement protocol

The timing instrumentation starts after each worker has moved its model replica
to its assigned GPU and synchronized. A barrier starts all workers together.
It stops after GPU synchronization and before moving replicas back to CPU.
Checkpoint loading, GPU transfer, unloading, and benchmark judging are
excluded. Prompt batching, token generation, ensemble scoring, all rejection
attempts, and forced acceptance are included.

The denominator is the exact count of final response tokens from `output_mask`.
Prompt and padding tokens are excluded. For response-level sampling, proposal
attempts remain in the timed numerator but only the accepted response is in the
denominator. The metric is therefore end-to-end multi-GPU wall time per final
response token, not single-request kernel latency.

The calibration uses `scripts/slurm/eval.slurm` on four GH200 GPUs in the
`debug` partition, `seed=42`, the first `ns` AlpacaEval2 prompts, a 512-token
prompt limit, and `mt=1024`. The 8-prompt smoke tests are only instrumentation
checks because they underutilize the batch size; use the 128-prompt jobs for
the comparison. The exact command overrides and job IDs are recorded below.

| ID | Configuration | Job ID | Result |
|---|---|---:|---|
| R-008-smoke-token | Tulu `L=4`, E4, token-level, `ns=8`, `mt=128` | 3458016 | 156.782 s / 802 final response tokens = 195.489 ms/token; instrumentation check only |
| R-008-smoke-response | Same, practical `mean_std`, `eta=.1`, `max_trials=16` | 3458017 | 74.760 s / 814 final response tokens = 91.843 ms/token; instrumentation check only |
| R-008-token-128 | Tulu `L=4`, E4, token-level, `ns=128`, `mt=1024` | 3458049 | 1009.018 s / 17,822 final response tokens = 56.616 ms/token |
| R-008-response-128 | Same, practical `mean_std`, `eta=.1`, `max_trials=16` | 3458050 | 680.878 s / 20,164 final response tokens = 33.767 ms/token; 1.18 proposals/prompt, 0% caps |
| R-008-token-e5 | Tulu `L=4`, E5, token-level, `ns=128`, `mt=1024` | 3458654 | 740.040 s / 18,494 final response tokens = 40.015 ms/token; output saved before teardown exit 7 |
| R-008-response-e5 | Same E5 setup, `generator_batch_size=32`, practical `mean_std`, `eta=.1`, `max_trials=64` | 3458835 | 795.820 s / 21,028 final response tokens = 37.846 ms/token; 1.07 proposals/prompt, 0% caps |
