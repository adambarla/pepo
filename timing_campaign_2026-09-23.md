# Full AlpacaEval latency campaign — 2026-09-23

## Purpose

Re-measure the two methods that are currently being compared using the same
full prompt set and the same execution settings. The older latency table is
not fully controlled: the 128-prompt jobs underfilled the workload, and the
response-level jobs used different `max_trials` and batch-size settings.

This is a controlled comparison, not a peak-throughput benchmark. A separate
batch-size/compile sweep can be run later if we want each method's maximum
hardware throughput.

## Fixed protocol

| Item | Setting |
|---|---|
| Model | `allenai/Llama-3.1-Tulu-3-8B-SFT` checkpoints |
| Ensemble | `L=4`, disjoint, `a=b=0.1`, shared backbone |
| Epochs | 1–6 |
| Prompts | all 805 `tatsu-lab/alpaca_eval` eval prompts (`ns=805`) |
| GPUs | 4 × NVIDIA GH200 120 GB, exclusive node |
| Dtype | bfloat16 |
| Prompt/response limits | `max_prompt_length=512`, `max_new_tokens=1024` |
| Generation batch | explicitly fixed to 32 per worker for both methods |
| Timing mode | `stop_after_generation=true`; Alpaca judging excluded |
| Existing responses | reused; `overwrite=false` |
| Token-level | top-p generator, temperature 1, top-p 0.9 |
| Mean–std RS | `eta=0.1`, `max_trials=128`, temperature 1, top-p 0.9 |
| Repetitions | 3 per method/epoch, seeds 1001/1002/1003 |
| Time limit | 6 hours per job |

`max_trials=128` is intentional: this campaign measures the requested full
rejection-sampling budget. It is not a claim that response-level sampling is
optimized for maximum throughput. The strict/min-RS method remains excluded;
its full-prompt extrapolations are tens of hours per epoch.

The timing metric is the generator's synchronized multi-GPU wall time divided
by the exact number of final response tokens. Model loading, GPU transfer,
unloading, and judging are excluded; proposal generation, ensemble scoring,
retries, and forced acceptance are included for RS.

The evaluator is configured with `overwrite=false`. If a matching response
file already exists in the repetition's output directory, it is reused rather
than regenerated. Such a reused job will not produce a new generation timing
line; the original job log remains the timing source.

## Reproducibility change

The evaluation entry point now calls `set_seed(cfg.seed)` before model
initialization. Each repetition has an explicit seed, and the same seed is
paired across the token-level and mean–std jobs. The remote checkout was
already a dirty working tree; the campaign uses that state plus this small
change, not a clean released commit. Current base revision: `363586b`.

## Submitted jobs

The submission script is
`scripts/schedule_timing_campaign_20260923.sh`.

The original token-level jobs (IDs 3499340–3499369 and their scores) were
canceled before they started because they used greedy decoding. The
mean–std jobs remain unchanged. The token-level arm is being replaced with
top-p decoding using `temperature=1`, `top_p=.9`, and the same seeds. The
replacement script is `scripts/resubmit_token_top_p_20260923.sh`.

| Repetition | Seed | Token-level top-p generation | Mean–std generation |
|---|---:|---:|---:|
| R1 | 1001 | 3502661–3502666 | 3499346–3499351 |
| R2 | 1002 | 3502667–3502672 | 3499358–3499363 |
| R3 | 1003 | 3502673–3502678 | 3499370–3499375 |

Scoring IDs are ordered by epoch E1 through E6. The old greedy token scores
were canceled with their generation jobs:

| Repetition | Token-level top-p scoring | Mean–std scoring (resubmitted) | Mean–std scoring (failed) |
|---|---|---|---|
| R1 | 3502679–3502684 | 3503286, 3503287, 3503288, 3503289, 3503290, 3503291 | 3499377, 3499379, 3499381, 3499383, 3499385, 3499387 |
| R2 | 3502685–3502690 | 3503292, 3503293, 3503294, 3503295, 3503296, 3503297 | 3499389, 3499393, 3499395, 3499397, 3499399, 3499401 |
| R3 | 3502691–3502696 | 3503298, 3503299, 3503300, 3503301, 3503302, 3503303 | 3499403, 3499405, 3499407, 3499409, 3499411, 3499413 |

Responses are isolated under:

```text
outputs/timing_campaign_20260923/{token,meanstd}_e{epoch}_r{rep}/
```

Slurm logs are under `outputs/slurm/eval/{jobid}.out` and `.err`. The key
lines to collect are `Generation timing`, `Proposal diagnostics` (RS), and
the final `Saved ... responses` line.

After each generation job, a dependent scoring job evaluates the response
file against the same full 805-response reference:

```text
outputs/alpaca_eval/responses/Llama-3.1-Tulu-3-8B-SFT-a0.0-b0.1-L1-e0_mt1024_responses.json
```

This produces raw and length-controlled AlpacaEval win-rates. Score outputs
are under `outputs/timing_campaign_20260923/scores/`. A score job reuses an
existing leaderboard in its output directory.

## Monitoring

```bash
squeue -u $USER -o "%.18i %.22j %.10T %.10M %.10l %.12P %R"
sacct -X -j 3499346-3502696 \
  --format=JobID,JobName%22,State,Elapsed,ExitCode,NodeList
grep -H "Generation timing\|Proposal diagnostics" outputs/slurm/eval/{3499346..3499375,3502661..3502678}.out
find outputs/slurm/score -type f -name "*.out" -print0 \\
  | xargs -0 grep -H "common805\|win_rate\|length_controlled"
```

## Expected duration

The completed fixed-batch full-prompt token top-p runs took roughly 2.3–2.8
hours. Mean–std with `max_trials=128` took roughly 0.7–1.0 hours in this
campaign; its cost rises with the number of proposals, especially at later
epochs. The strict/min-RS `max_trials=128` runs are still not a realistic
full-prompt timing target.

## Generation results (2026-09-24)

All 36 generation jobs completed successfully and saved 805 responses. Values
below are mean ± sample standard deviation over the three repetitions:

| Epoch | Token top-p wall time | Token top-p ms/final token | Mean–std RS wall time | Mean–std RS ms/final token | RS trials/prompt |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.44 h | 40.06 ± 0.33 | 0.72 h | 11.28 ± 0.46 | 1.16 |
| 2 | 2.53 h | 43.29 ± 5.09 | 0.73 h | 12.34 ± 0.55 | 1.19 |
| 3 | 2.62 h | 46.21 ± 3.05 | 0.70 h | 11.91 ± 0.57 | 1.32 |
| 4 | 2.52 h | 43.53 ± 2.74 | 0.77 h | 13.06 ± 0.18 | 1.76 |
| 5 | 2.59 h | 43.64 ± 0.45 | 0.78 h | 12.91 ± 0.39 | 2.24 |
| 6 | 2.77 h | 43.61 ± 0.41 | 0.90 h | 14.25 ± 0.56 | 3.41 |

The RS trial count rises sharply after E3, especially at E6, but remains well
below the 128-trial cap. The first response-level scoring submissions failed
because they started before the response files were visible to the scoring
job. All response files are now present and contain 805 records. Replacement
score jobs 3503286–3503303 are queued; top-p score jobs 3502679–3502696 are
also queued. No new win-rate values are available yet.

## Interpretation rule

Use the mean and standard deviation over the three repetitions for the primary
timing table, and report raw plus length-controlled win-rate for each
method/epoch. Report the exact settings above alongside the result. Do not mix
these values with the old 128-prompt numbers or with runs using
`max_trials=16/64`; the current response-level campaign intentionally uses
`max_trials=128`.
