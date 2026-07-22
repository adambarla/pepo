#
# This notebook provides analysis and visualization for PEPO (Preference
# Ensemble Policy Optimization) experiments.

# %% [markdown]
# ## Setup
#
# Import necessary libraries and the `utils` module.

# %%
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    ipython = get_ipython()  # type: ignore[name-defined]
except NameError:
    ipython = None

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

REPO_ROOT = Path.cwd()
load_dotenv(REPO_ROOT / ".env")

# Append cwd/notebooks to sys.path.
sys.path.append(str(REPO_ROOT / "notebooks"))

import utils  # noqa: E402

MODELS = utils.MODELS
# 7/8B models only (exclude Yi-34B-Chat).
MODELS_78B = MODELS[:3]
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Data Loading
#
# Fetch runs from WandB or load from local cache.

# %%
# Set force_refresh=True to fetch new runs from WandB.
df = utils.get_runs_df(force_refresh=False)
print(f"Loaded {len(df)} runs.")

# %% [markdown]
# # Experiment 1: Greedy Sampling (AlpacaEval)
#
# Comparing various algorithms (DPO, SFT+DPO, $\chi^2$PO, PEPO) using
# token-level greedy sampling.

# %%
# Prepare data for all models.
dfs = [utils.get_exp1_data(df, model_idx=i) for i in range(len(MODELS))]

# TODO: Mistral epoch 5 is an outlier on AlpacaEval — investigate why.
MISTRAL_IDX = 2
mask = (dfs[MISTRAL_IDX]["epoch"] == 5) & (dfs[MISTRAL_IDX]["algorithm"] == "chi2po")
dfs[MISTRAL_IDX] = dfs[MISTRAL_IDX][~mask]

dfs_78b = dfs[: len(MODELS_78B)]
for model, d in zip(MODELS, dfs):
    counts = d["algorithm"].value_counts().to_dict()
    print(f"{model.split('/')[-1]}: {len(d)} runs, {counts}")

# %% [markdown]
# ### Best PEPO $L$ vs. Other Algorithms (7/8B Models)
#
# For each model, keep only the best-performing PEPO $L$ variant and compare
# it against DPO, SFT+DPO and $\chi^2$PO.

# %%
best_dfs = [utils.keep_best_pepo_L(d, y_col="winrate_initial") for d in dfs_78b]

utils.plot_multi_model_comparison(
    best_dfs,
    exclude_algos=[],
    x_col="epoch",
    y_col="winrate_initial",
    se_col="standard_error_initial",
    save_path=str(FIGURES_DIR / "alpaca_win_rate_methods.pdf"),
)

# %% [markdown]
# ### Length-Controlled Win Rate (7/8B Models)
#
# Same comparison as above, but using the length-controlled win rate to
# correct for verbosity bias.

# %%
best_lc_dfs = [utils.keep_best_pepo_L(d, y_col="winrate_initial_lc") for d in dfs_78b]

utils.plot_multi_model_comparison(
    best_lc_dfs,
    exclude_algos=[],
    x_col="epoch",
    y_col="winrate_initial_lc",
    se_col="standard_error_initial_lc",
    save_path=str(FIGURES_DIR / "win_rate_initial_lc_best_pepo.pdf"),
)

# %% [markdown]
# ### DPO vs. PEPO $L$ Variants
#
# Compare DPO against the different PEPO $L$ values for each model.

# %%
utils.plot_multi_model_comparison(
    dfs,
    exclude_algos=["sftdpo", "chi2po"],
    x_col="epoch",
    y_col="winrate_initial",
    se_col="standard_error_initial",
    save_path=str(FIGURES_DIR / "alpaca_win_rate_L_ablation.pdf"),
)

# %% [markdown]
# # MT-Bench Win Rate
#
# Plot MT-Bench win rate against each model's initial checkpoint.
# Use the tie-adjusted win rate so the epoch-0 initial self-comparison is 50%.

# %%
mtbench_dfs = [utils.get_mtbench_data(df, model_idx=i) for i in range(len(MODELS))]
mtbench_dfs_78b = mtbench_dfs[: len(MODELS_78B)]

# %% [markdown]
# ### Best PEPO $L$ vs. Other Algorithms (7/8B Models)
#
# For each model, keep only the best-performing PEPO $L$ variant and compare
# it against DPO, SFT+DPO and $\chi^2$PO.

# %%
mtbench_best_dfs = [
    utils.keep_best_pepo_L(d, y_col="mtbench_winrate_adjusted") for d in mtbench_dfs_78b
]

utils.plot_multi_model_comparison(
    mtbench_best_dfs,
    exclude_algos=[],
    x_col="epoch",
    y_col="mtbench_winrate_adjusted",
    save_path=str(FIGURES_DIR / "mtbench_win_rate_methods.pdf"),
)

# %% [markdown]
# ### DPO vs. PEPO $L$ Variants
#
# Compare DPO against the different PEPO $L$ values for each model.

# %%
utils.plot_multi_model_comparison(
    mtbench_dfs,
    exclude_algos=["sftdpo", "chi2po"],
    x_col="epoch",
    y_col="mtbench_winrate_adjusted",
    save_path=str(FIGURES_DIR / "mtbench_win_rate_L_ablation.pdf"),
)

# %% [markdown]
# ### Raw MT-Bench Scores
#
# Average judge score (0-10 scale) over epochs. Higher is better.

# %%
mtbench_best_score_dfs = [
    utils.keep_best_pepo_L(d, y_col="mtbench_score") for d in mtbench_dfs_78b
]

utils.plot_multi_model_comparison(
    mtbench_best_score_dfs,
    exclude_algos=[],
    x_col="epoch",
    y_col="mtbench_score",
    save_path=str(FIGURES_DIR / "mtbench_score_methods.pdf"),
)

# %%
utils.plot_multi_model_comparison(
    mtbench_dfs,
    exclude_algos=["sftdpo", "chi2po"],
    x_col="epoch",
    y_col="mtbench_score",
    save_path=str(FIGURES_DIR / "mtbench_score_L_ablation.pdf"),
)

# %%
# Combined main figure with all 4 panels.
utils.plot_main_figure(
    best_dfs,
    dfs_78b,
    mtbench_best_dfs,
    mtbench_dfs_78b,
    save_path=str(FIGURES_DIR / "main_figure.pdf"),
)

# %% [markdown]
# ## Summary Tables

# %%
# AlpacaEval: best win rate per model per algorithm.
alpaca_summary = utils.get_best_winrates(
    dfs,
    y_col="winrate_initial",
    se_col="standard_error_initial",
    aggregate_pepo=True,
)
print("=== AlpacaEval ===")
latex = utils.format_winrates_latex(alpaca_summary, pivot=True)
print(latex)
(FIGURES_DIR / "table_alpaca.tex").write_text(latex)

# %%
# MT-Bench: best win rate per model per algorithm.
mtbench_summary = utils.get_best_winrates(
    mtbench_dfs,
    y_col="mtbench_winrate_adjusted",
    se_col=None,
    aggregate_pepo=True,
)
print("=== MT-Bench ===")
latex = utils.format_winrates_latex(mtbench_summary, pivot=True, include_se=False)
print(latex)
(FIGURES_DIR / "table_mtbench.tex").write_text(latex)

# %% [markdown]
# # Experiment 2: Rejection Sampling vs Token-Level
#
# Comparing performance across different sampling strategies (Greedy vs.
# Rejection Sampling variants).

# %%
# Get data for Experiment 2 (epochs 1-5).
df_exp2 = utils.get_exp2_data(df, model_idx=0, epoch_range=(1, 5))

print("Algorithms found:", sorted(df_exp2["algorithm"].dropna().unique().tolist()))

# Plot individual variants.
utils.plot_multi_model_comparison(
    [df_exp2],
    aggregate_best=False,
    se_col="standard_error_initial",
)

# Plot aggregated (Best Rejection vs Best Token-Level).
utils.plot_multi_model_comparison(
    [df_exp2],
    aggregate_best=True,
    se_col="standard_error_initial",
)

# %% [markdown]
# ### Rejection Sampling vs Token-Level Comparison Table

# %%
# Comparison table for epochs 1-3.
comp_table = utils.format_exp2_comparison_table(
    df_exp2,
    epochs=[1, 2, 3],
    aggregate_best=True,
)
print(comp_table)
(FIGURES_DIR / "table_exp2_rej_vs_token_epoch.tex").write_text(comp_table)
print("Saved to figures/table_exp2_rej_vs_token_epoch.tex")
