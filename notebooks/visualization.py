# %%
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import utils
from dotenv import load_dotenv

try:
    ipython = get_ipython()
except NameError:
    ipython = None

REPO_ROOT = Path.cwd()
load_dotenv(REPO_ROOT / ".env")
sys.path.append(str(REPO_ROOT / "notebooks"))

if ipython is None:
    matplotlib.use("Agg")

MODELS = utils.MODELS
MODELS_78B = MODELS[:3]
try:
    HERE = Path(__file__).resolve()
    FIGURES_DIR = HERE.parents[2] / "figures"
except NameError:
    FIGURES_DIR = Path.cwd().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Data Loading
df = utils.get_runs_df(force_refresh=False)
print(f"Loaded {len(df)} runs.")
BEST_L = utils.find_best_global_L(df)
print(f"Global best PEPO L = {BEST_L}")

# %%
# Exp1: AlpacaEval
dfs = [utils.get_exp1_data(df, i) for i in range(len(MODELS))]
mask = (dfs[2]["epoch"] == 5) & (dfs[2]["algorithm"] == "chi2po")
dfs[2] = dfs[2][~mask]
dfs_78b = dfs[:3]
for m, d in zip(MODELS, dfs):
    sn = m.split("/")[-1]
    print(f"{sn}: {len(d)} runs, {d['algorithm'].value_counts().to_dict()}")


def kbp(d, y, f):
    return utils.keep_best_pepo_L(d, y_col=y, fixed_L=f)


best_dfs = [kbp(d, "winrate_initial", BEST_L) for d in dfs]

# Best PEPO L vs other algorithms (7/8B)
utils.plot_multi_model_comparison(
    best_dfs,
    x_col="epoch",
    y_col="winrate_initial",
    se_col="standard_error_initial",
    exclude_algos=[],
    save_path=str(FIGURES_DIR / "alpaca_win_rate_methods.pdf"),
)

# DPO vs PEPO L Variants
utils.plot_multi_model_comparison(
    dfs,
    x_col="epoch",
    y_col="winrate_initial",
    se_col="standard_error_initial",
    exclude_algos=["sftdpo", "chi2po"],
    save_path=str(FIGURES_DIR / "alpaca_win_rate_L_ablation.pdf"),
)

# %%
# MT-Bench
mt_df = [utils.get_mtbench_data(df, i) for i in range(len(MODELS))]
mt_df_78b = mt_df[:3]

mt_best = [kbp(d, "mtbench_winrate_adjusted", BEST_L) for d in mt_df_78b]

utils.plot_multi_model_comparison(
    mt_best,
    x_col="epoch",
    y_col="mtbench_winrate_adjusted",
    exclude_algos=[],
    save_path=str(FIGURES_DIR / "mtbench_win_rate_methods.pdf"),
)
utils.plot_multi_model_comparison(
    mt_df,
    x_col="epoch",
    y_col="mtbench_winrate_adjusted",
    exclude_algos=["sftdpo", "chi2po"],
    save_path=str(FIGURES_DIR / "mtbench_win_rate_L_ablation.pdf"),
)


try:
    utils.plot_main_figure(
        best_dfs[:3],
        dfs_78b,
        mt_best,
        mt_df_78b,
        save_path=str(FIGURES_DIR / "main_figure.pdf"),
    )
except (IndexError, ValueError):
    print("Skipping main_figure plot (no data).")


# %%
# Summary tables
def _table(name, data, y_col, se_col, write_best=True, write_final=True):
    fl = [kbp(d, y_col, BEST_L) for d in data]
    inc = se_col is not None
    if write_best:
        s = utils.get_best_winrates(fl, y_col=y_col, se_col=se_col, aggregate_pepo=True)
        latex = utils.format_winrates_latex(s, pivot=True, include_se=inc)
        print(f"=== {name.title()} (L=3) ===")
        print(latex)
        (FIGURES_DIR / f"table_{name}_bestepoch.tex").write_text(latex)
    if write_final:
        f = utils.get_final_epoch_winrates(
            data,
            y_col=y_col,
            se_col=se_col,
            fixed_L=BEST_L,
        )
        latex = utils.format_winrates_latex(f, pivot=True, include_se=inc)
        print(f"=== {name.title()} (Final Epoch) ===")
        print(latex)
        (FIGURES_DIR / f"table_{name}_final_epoch.tex").write_text(latex)


_table("alpaca", dfs, "winrate_initial", "standard_error_initial")
_table("mtbench", mt_df, "mtbench_winrate_adjusted", None)


# %%
# Margin analysis
def _sub(mid):
    mask = (
        (df["config/model/model_id"] == mid)
        | (df["config/backbone/model_id"] == mid)
        | (df["config/model/backbone/model_id"] == mid)
    )
    sub = df[mask]
    c = "config/model/generator/greedy_sampling"
    if c in sub.columns:
        sub = sub[sub[c].map(lambda v: v is True or str(v).lower() == "true")]
    return sub[
        sub["name"].str.contains("eval", na=False)
        & ~sub["name"].str.contains("mtbench", na=False)
    ]


rows = []
for i in range(3):
    mid = utils.MODELS[i]
    sub = _sub(mid)
    short = mid.rsplit("/", 1)[-1]
    wr_col = f"summary/eval/tatsu-lab/alpaca_eval/mt1024/{short}-a0.0-b0.1-L1/win_rate"
    se_col = (
        f"summary/eval/tatsu-lab/alpaca_eval/mt1024/{short}-a0.0-b0.1-L1/standard_error"
    )
    for algo_label, L, a in [("DPO", 1, 0.0), ("Margin", 1, 0.1), ("PEPO", 3, 0.1)]:
        runs = sub[
            (sub["config/L"] == L)
            & (sub["config/model/alpha"] == a)
            & sub[wr_col].notna()
        ]
        if runs.empty:
            continue
        best = runs.loc[runs[wr_col].idxmax()]
        se_val = best.get(se_col)
        rows.append(
            {
                "model": short,
                "algorithm": algo_label,
                "best_winrate": best[wr_col],
                "best_epoch": best.get("summary/eval/epoch"),
                "standard_error": se_val if pd.notna(se_val) else None,
            }
        )

margin_summary = pd.DataFrame(rows)
lt = utils.format_winrates_latex(
    margin_summary,
    pivot=True,
    include_se=True,
    short_names=True,
)
print("=== Margin Experiment: Best Win Rate ===")
print(lt)
(FIGURES_DIR / "table_margin_bestepoch.tex").write_text(lt)

# %%
# Method x epochs (Tulu-3-8B rebuttal)
mid = utils.MODELS[0]
sub = _sub(mid)
short = mid.rsplit("/", 1)[-1]
wr_col = f"summary/eval/tatsu-lab/alpaca_eval/mt1024/{short}-a0.0-b0.1-L1/win_rate"
epochs = list(range(0, 17, 2))
methods = [
    ("DPO (L=1)", 1, 0.0),
    ("Margin (L=1)", 1, 0.1),
    ("PEPO L=2", 2, 0.1),
    ("PEPO L=3", 3, 0.1),
    ("PEPO L=4", 4, 0.1),
]
print(f"\n% {short}")
print("\\begin{tabular}{l" + "c" * len(epochs) + "}")
print("\\toprule")
print("Method & " + " & ".join(f"E{ep}" for ep in epochs) + " \\\\")
print("\\midrule")
for label, L, a in methods:
    vals = []
    for ep in epochs:
        r = sub[
            (sub["config/L"] == L)
            & (sub["config/model/alpha"] == a)
            & (sub["summary/eval/epoch"] == float(ep))
        ]
        vals.append(
            f"{float(r.iloc[0][wr_col]):.1f}"
            if not r.empty and pd.notna(r.iloc[0].get(wr_col))
            else "---"
        )
    print(label + " & " + " & ".join(vals) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# %%
# Disjoint vs bootstrap (Tulu-3-8B)
mid = utils.MODELS[0]
sub = _sub(mid)
short = mid.rsplit("/", 1)[-1]
sub = sub[(sub["config/L"] == 4) & (sub["config/model/alpha"] == 0.1)]
wr_col = f"summary/eval/tatsu-lab/alpaca_eval/mt1024/{short}-a0.0-b0.1-L1/win_rate"
epochs = list(range(0, 17, 2))
print(f"\n% {short}")
print("\\begin{tabular}{l" + "c" * len(epochs) + "}")
print("\\toprule")
print("Method & " + " & ".join(f"E{ep}" for ep in epochs) + " \\\\")
print("\\midrule")
is_boot = sub["name"].str.contains("bootstrap", na=False)
for label, mask in [("PEPO L=4 disjoint", ~is_boot), ("PEPO L=4 bootstrap", is_boot)]:
    vals = []
    for ep in epochs:
        r = sub[mask & (sub["summary/eval/epoch"] == float(ep)) & sub[wr_col].notna()]
        vals.append(f"{float(r.iloc[0][wr_col]):.1f}" if not r.empty else "---")
    print(label + " & " + " & ".join(vals) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")
