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
from IPython.display import display

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

# %% [markdown]
# ## Data Loading
#
# Fetch runs from WandB or load from local cache.

# %%
# Set force_refresh=True to fetch new runs from WandB.
df = utils.get_runs_df(force_refresh=False)
print(f"Loaded {len(df)} runs.")

# %% [markdown]
# # Experiment 1: Greedy Sampling
#
# Comparing various algorithms (DPO, SFT+DPO, $\chi^2$PO, PEPO) using
# token-level greedy sampling.

# %%
# Prepare data for all 4 models.
dfs = [utils.get_exp1_data(df, model_idx=i) for i in range(len(MODELS))]

# Plot comparison with error bands.
utils.plot_multi_model_comparison(
    dfs,
    exclude_algos=["sftdpo", "chi2po"],
    x_col="epoch",
    y_col="winrate_initial",
    se_col="standard_error_initial",
    save_path="figures/win_rate_initial.pdf",
)

# %% [markdown]
# # MT-Bench Win Rate
#
# Plot MT-Bench win rate against each model's initial checkpoint.

# %%
# Use tie-adjusted win rate so the epoch-0 initial self-comparison is 50%.
mtbench_dfs = [utils.get_mtbench_data(df, model_idx=i) for i in range(len(MODELS))]

utils.plot_multi_model_comparison(
    mtbench_dfs,
    exclude_algos=[],
    x_col="epoch",
    y_col="mtbench_winrate_adjusted",
    save_path="figures/mtbench_win_rate_adjusted.pdf",
)

# %% [markdown]
# ### Best Win Rates Summary Table (LaTeX)

# %%
# Get summary of best win rates.
summary_df = utils.get_best_winrates(dfs, aggregate_pepo=True)

# Format as LaTeX table.
latex_table = utils.format_winrates_latex(summary_df, pivot=True)
print("LaTeX Output:")
print(latex_table)

display(summary_df.head())

# %% [markdown]
# # Experiment 2: Rejection Sampling vs Token-Level
#
# Comparing performance across different sampling strategies (Greedy vs.
# Rejection Sampling variants).

# %%
# Get data for Experiment 2 (epochs 1-5).
df_exp2 = utils.get_exp2_data(df, model_idx=0, epoch_range=(1, 5))

print("Algorithms found:", df_exp2["algorithm"].unique())

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
# Generate comparison table for epochs 1-3.
comp_table = utils.format_exp2_comparison_table(
    df_exp2,
    epochs=[1, 2, 3],
    aggregate_best=True,
)
print(comp_table)
