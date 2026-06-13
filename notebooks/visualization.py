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
<<<<<<< HEAD
=======
from IPython.display import display
>>>>>>> ade9815 (chore: visualization notebook as script (repl in zed IDE))

try:
    ipython = get_ipython()  # type: ignore[name-defined]
except NameError:
    ipython = None

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
<<<<<<< HEAD
else:
    import matplotlib

    matplotlib.use("Agg")
=======
>>>>>>> ade9815 (chore: visualization notebook as script (repl in zed IDE))

REPO_ROOT = Path.cwd()
load_dotenv(REPO_ROOT / ".env")

# Append cwd/notebooks to sys.path.
sys.path.append(str(REPO_ROOT / "notebooks"))

import utils  # noqa: E402

MODELS = utils.MODELS
<<<<<<< HEAD
# 7/8B models only (exclude Yi-34B-Chat).
MODELS_78B = MODELS[:3]
try:
    HERE = Path(__file__).resolve()
    FIGURES_DIR = HERE.parents[2] / "figures"
except NameError:
    FIGURES_DIR = Path.cwd().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
=======
>>>>>>> ade9815 (chore: visualization notebook as script (repl in zed IDE))

# %% [markdown]
# ## Data Loading
#
# Fetch runs from WandB or load from local cache.

# %%
# Set force_refresh=True to fetch new runs from WandB.
df = utils.get_runs_df(force_refresh=False)
print(f"Loaded {len(df)} runs.")

<<<<<<< HEAD
# Compute global best L once, use for all figures and tables.
BEST_L = utils.find_best_global_L(df)
print(f"Global best PEPO L = {BEST_L}")

# %% [markdown]
# # Experiment 1: Greedy Sampling (AlpacaEval)
=======
# %% [markdown]
# # Experiment 1: Greedy Sampling
>>>>>>> ade9815 (chore: visualization notebook as script (repl in zed IDE))
#
# Comparing various algorithms (DPO, SFT+DPO, $\chi^2$PO, PEPO) using
# token-level greedy sampling.

# %%
<<<<<<< HEAD
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
best_dfs = [
    utils.keep_best_pepo_L(d, y_col="winrate_initial", fixed_L=BEST_L) for d in dfs
]

utils.plot_multi_model_comparison(
    best_dfs,
    exclude_algos=[],
    x_col="epoch",
    y_col="winrate_initial",
    se_col="standard_error_initial",
    save_path=str(FIGURES_DIR / "alpaca_win_rate_methods.pdf"),
)

# %% [markdown]
# ### DPO vs. PEPO $L$ Variants
#
# Compare DPO against the different PEPO $L$ values for each model.

# %%
=======
# Prepare data for all 4 models.
dfs = [utils.get_exp1_data(df, model_idx=i) for i in range(len(MODELS))]

# Plot comparison with error bands.
>>>>>>> ade9815 (chore: visualization notebook as script (repl in zed IDE))
utils.plot_multi_model_comparison(
    dfs,
    exclude_algos=["sftdpo", "chi2po"],
    x_col="epoch",
    y_col="winrate_initial",
    se_col="standard_error_initial",
<<<<<<< HEAD
    save_path=str(FIGURES_DIR / "alpaca_win_rate_L_ablation.pdf"),
=======
    save_path="figures/win_rate_initial.pdf",
>>>>>>> ade9815 (chore: visualization notebook as script (repl in zed IDE))
)

# %% [markdown]
# # MT-Bench Win Rate
#
# Plot MT-Bench win rate against each model's initial checkpoint.
<<<<<<< HEAD
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
    utils.keep_best_pepo_L(d, y_col="mtbench_winrate_adjusted", fixed_L=BEST_L)
    for d in mtbench_dfs_78b
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
    utils.keep_best_pepo_L(d, y_col="mtbench_score", fixed_L=BEST_L)
    for d in mtbench_dfs_78b
]
mtbench_best_score_dfs = [d for d in mtbench_best_score_dfs if not d.empty]

if mtbench_best_score_dfs:
    try:
        utils.plot_multi_model_comparison(
            mtbench_best_score_dfs,
            exclude_algos=[],
            x_col="epoch",
            y_col="mtbench_score",
            save_path=str(FIGURES_DIR / "mtbench_score_methods.pdf"),
        )
    except (IndexError, ValueError):
        print("Skipping mtbench_score_methods plot (no data).")

# %%
try:
    utils.plot_multi_model_comparison(
        mtbench_dfs,
        exclude_algos=["sftdpo", "chi2po"],
        x_col="epoch",
        y_col="mtbench_score",
        save_path=str(FIGURES_DIR / "mtbench_score_L_ablation.pdf"),
    )
except (IndexError, ValueError):
    print("Skipping mtbench_score_L_ablation plot (no data).")

# %%
# Combined main figure with all 4 panels.
try:
    utils.plot_main_figure(
        best_dfs,
        dfs_78b,
        mtbench_best_dfs,
        mtbench_dfs_78b,
        save_path=str(FIGURES_DIR / "main_figure.pdf"),
    )
except (IndexError, ValueError):
    print("Skipping main_figure plot (no data).")

# %% [markdown]
# ## Summary Tables

# %%
# AlpacaEval: best win rate per model per algorithm — fixed PEPO L=3 (no oracle over L).
dfs_fixedL = [
    utils.keep_best_pepo_L(d, y_col="winrate_initial", fixed_L=BEST_L) for d in dfs
]
alpaca_summary = utils.get_best_winrates(
    dfs_fixedL,
    y_col="winrate_initial",
    se_col="standard_error_initial",
    aggregate_pepo=True,
)
print("=== AlpacaEval (L=3) ===")
latex = utils.format_winrates_latex(alpaca_summary, pivot=True)
print(latex)
(FIGURES_DIR / "table_alpaca_bestepoch.tex").write_text(latex)

# %%
# MT-Bench: best win rate per model per algorithm — fixed PEPO L=3.
mtbench_dfs_fixedL = [
    utils.keep_best_pepo_L(d, y_col="mtbench_winrate_adjusted", fixed_L=BEST_L)
    for d in mtbench_dfs
]
mtbench_summary = utils.get_best_winrates(
    mtbench_dfs_fixedL,
    y_col="mtbench_winrate_adjusted",
    se_col=None,
    aggregate_pepo=True,
)
print("=== MT-Bench (L=3) ===")
latex = utils.format_winrates_latex(mtbench_summary, pivot=True, include_se=False)
print(latex)
# save
(FIGURES_DIR / "table_mtbench_bestepoch.tex").write_text(latex)

# %% [markdown]
# ## Final-Epoch (Non-Oracle) Tables
#
# Win rates at the last epoch available per algorithm, rather than the best epoch.

# %%
# AlpacaEval: final epoch win rate per model per algorithm.
alpaca_final = utils.get_final_epoch_winrates(
    dfs, y_col="winrate_initial", se_col="standard_error_initial", fixed_L=BEST_L
)
print("=== AlpacaEval (Final Epoch) ===")
latex = utils.format_winrates_latex(alpaca_final, pivot=True)
print(latex)
(FIGURES_DIR / "table_alpaca_final_epoch.tex").write_text(latex)

# %%
# MT-Bench: final epoch win rate per model per algorithm.
mtbench_final = utils.get_final_epoch_winrates(
    mtbench_dfs, y_col="mtbench_winrate_adjusted", se_col=None, fixed_L=BEST_L
)
print("=== MT-Bench (Final Epoch) ===")
latex = utils.format_winrates_latex(mtbench_final, pivot=True, include_se=False)
print(latex)
(FIGURES_DIR / "table_mtbench_final_epoch.tex").write_text(latex)
=======

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
>>>>>>> ade9815 (chore: visualization notebook as script (repl in zed IDE))
