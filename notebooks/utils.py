import ast
import os
import re

import pandas as pd
import wandb

MODELS = [
    "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "alignment-handbook/zephyr-7b-sft-full",
    "HuggingFaceH4/mistral-7b-sft-beta",
    "01-ai/Yi-34B-Chat",
]


def _flatten(d, pk=""):
    r = {}
    for k, v in d.items():
        nk = f"{pk}/{k}" if pk else k
        if isinstance(v, dict):
            r.update(_flatten(v, nk))
        else:
            r[nk] = v
    return r


def get_runs_df(
    entity=None,
    project="pepo",
    filters=None,
    cache_path=None,
    force_refresh=False,
    since=None,
):
    entity = entity or os.getenv("WANDB_ENTITY") or "pepo-team"
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), ".cache", "runs_cache.csv")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    df_cached = None
    if os.path.exists(cache_path):
        df_cached = pd.read_csv(cache_path, low_memory=False)
        if "created_at" in df_cached.columns:
            df_cached["created_at"] = pd.to_datetime(df_cached["created_at"])

    if since is None and not force_refresh and df_cached is not None:
        return df_cached

    if (
        since is None
        and not force_refresh
        and df_cached is not None
        and "created_at" in df_cached.columns
    ):
        since = df_cached["created_at"].max()

    if since is not None:
        st = pd.Timestamp(since)
        tf = {"created_at": {"$gt": st.isoformat()}}
        filters = {"$and": [filters, tf]} if filters else tf

    api = wandb.Api()
    data = []
    for run in api.runs(path=f"{entity}/{project}", filters=filters):
        e = {
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
        }
        for k, v in _flatten(run.config, "config").items():
            e[k] = str(v) if isinstance(v, (dict, list, tuple)) else v
        for k, v in _flatten(run.summary, "summary").items():
            e[k] = str(v) if isinstance(v, (dict, list, tuple)) else v
        data.append(e)
    df_new = pd.DataFrame(data)
    if df_new.empty and df_cached is not None and not force_refresh:
        return df_cached

    if df_cached is not None and not force_refresh:
        df = pd.concat([df_cached, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["run_id"], keep="last")
    else:
        df = df_new
    df.to_csv(cache_path, index=False)
    return df


def _mod(df, mid):
    return df[
        (df["config/model/model_id"] == mid)
        | (df["config/backbone/model_id"] == mid)
        | (df["config/model/backbone/model_id"] == mid)
    ]


def _alp(df):
    return df[
        df["name"].str.contains("eval", na=False)
        & ~df["name"].str.contains("mtbench", na=False)
    ]


def _grd(df):
    c = "config/model/generator/greedy_sampling"
    if c not in df.columns:
        return df
    return df[df[c].map(lambda v: v is True or str(v).lower() == "true")]


def _spl(df):
    for c in (
        "config/model/split_mode",
        "config/dataset/split_mode",
        "config/split_mode",
    ):
        if c in df.columns:
            m = df[c].astype(str).str.lower()
            df = df[m.isin(["nan", "none", "", "disjoint"]) | df[c].isna()]
    return df


def _dedup(df, cols):
    if df.empty or not all(c in df.columns for c in cols):
        return df
    df = df.copy()
    return df.sort_values("created_at").groupby(cols, as_index=False).last()


def _pepo_wr(df):
    p = "summary/eval/tatsu-lab/alpaca_eval/mt1024/"
    for c, n in [
        ("win_rate", "winrate_gpt4"),
        ("standard_error", "standard_error_gpt4"),
    ]:
        if f"{p}{c}" in df.columns:
            df[n] = df[f"{p}{c}"]
    return df


def _init_wr(df, mid, lc=True):
    s = mid.split("/")[-1]
    p = f"summary/eval/tatsu-lab/alpaca_eval/mt1024/{s}-a0.0-b0.1-L1/"
    for c, n in [
        ("win_rate", "winrate_initial"),
        ("standard_error", "standard_error_initial"),
    ]:
        if f"{p}{c}" in df.columns:
            df[n] = df[f"{p}{c}"]
    if lc:
        for c, n in [
            ("length_controlled_winrate", "winrate_initial_lc"),
            ("lc_standard_error", "standard_error_initial_lc"),
        ]:
            if f"{p}{c}" in df.columns:
                df[n] = df[f"{p}{c}"]
        if "winrate_initial_lc" in df.columns:
            df.loc[df["epoch"] == 0, "winrate_initial_lc"] = 50.0
    if "epoch" in df.columns:
        df.loc[df["epoch"] == 0, "winrate_initial"] = 50.0
    return df


def _ins_epoch0(df, ac, ec, wc, lc=None):
    nr = []
    for a in df[ac].dropna().unique():
        if (df[df[ac] == a][ec] == 0).any():
            continue
        b = df[df[ac] == a].iloc[0].copy()
        b[ec], b[wc] = 0, 50.0
        if lc and lc in df.columns:
            b[lc] = 50.0
        nr.append(b)
    return pd.concat([df, pd.DataFrame(nr)], ignore_index=True) if nr else df


def get_exp1_data(df, model_idx=0):
    mid = MODELS[model_idx]
    df = _mod(df, mid)
    df = df[~((df["config/L"] == 1) & (df["config/model/alpha"] == 0.1))]
    if "name" in df.columns:
        df = df[~df["name"].astype(str).str.contains("bootstrap", na=False)]
    df = _spl(df)
    df = _grd(df)
    df = _alp(df)
    if "config/L" in df.columns:
        df["L"] = df["config/L"]
    if "summary/eval/epoch" in df.columns:
        df["epoch"] = df["summary/eval/epoch"]
    df["model"] = mid
    if "config/model/wandb/tags" in df.columns:
        df["config/model/wandb/tags"] = df["config/model/wandb/tags"].apply(
            lambda v: (
                ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else v
            )
        )

        def _ga(row):
            t = row["config/model/wandb/tags"]
            if isinstance(t, list) and t:
                return t[0]
            return f"pepo-L{row['L']}" if "L" in row else "unknown"

        df["algorithm"] = df.apply(_ga, axis=1)
    if "algorithm" in df.columns and "epoch" in df.columns:
        df = _dedup(df, ["algorithm", "epoch"])
    df = _pepo_wr(df)
    df = _init_wr(df, mid)
    if (
        "algorithm" in df.columns
        and "epoch" in df.columns
        and "winrate_initial" in df.columns
    ):
        df = _ins_epoch0(
            df, "algorithm", "epoch", "winrate_initial", "winrate_initial_lc"
        )
    return df


def get_mtbench_data(df, model_idx=0):
    mid = MODELS[model_idx]
    df = _mod(df, mid)
    df = df[df["name"].str.contains("mtbench-eval", na=False)].copy()
    p = "summary/eval/mt_bench/mt1024/"
    for c, n in [
        ("win_rate", "mtbench_winrate"),
        ("win_rate_adjusted", "mtbench_winrate_adjusted"),
    ]:
        if f"{p}{c}" in df.columns:
            df[n] = df[f"{p}{c}"] * 100.0
    if "summary/eval/epoch" in df.columns:
        df["epoch"] = df["summary/eval/epoch"]
    if "config/L" in df.columns:
        df["L"] = df["config/L"]
    df["model"] = mid
    if "config/model/wandb/tags" in df.columns:
        df["config/model/wandb/tags"] = df["config/model/wandb/tags"].apply(
            lambda v: (
                ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else v
            )
        )

    def _algo_from_tags_or_l(row):
        t = row.get("config/model/wandb/tags")
        if isinstance(t, list) and t:
            return t[0]
        L = row.get("L", row.get("config/L"))
        return f"pepo-L{int(L)}" if pd.notna(L) else "unknown"

    df["algorithm"] = df.apply(_algo_from_tags_or_l, axis=1)
    if "algorithm" in df.columns and "epoch" in df.columns:
        df = _dedup(df, ["algorithm", "epoch"])
    if "epoch" in df.columns and "mtbench_winrate_adjusted" in df.columns:
        df.loc[df["epoch"] == 0, "mtbench_winrate_adjusted"] = 50.0
    if (
        "algorithm" in df.columns
        and "epoch" in df.columns
        and "mtbench_winrate_adjusted" in df.columns
    ):
        df = _ins_epoch0(df, "algorithm", "epoch", "mtbench_winrate_adjusted")
    return df


def get_exp2_data(df, model_idx=0, epoch_range=None):
    mid = MODELS[model_idx]
    df = _mod(df, mid)
    df = _alp(df)

    def _is_bon(r):
        return "bon" in str(r.get("name", "")).lower()

    if "config/L" in df.columns:
        is_pepo = df["config/L"] >= 2
        df = df[is_pepo | df.apply(_is_bon, axis=1)].copy()
        df["L"] = df["config/L"]
    else:
        df = df.copy()
    if "summary/eval/epoch" in df.columns:
        df["epoch"] = df["summary/eval/epoch"]
    df["model"] = mid

    def _ss(row):
        L = row.get("L", row.get("config/L"))
        name = str(row.get("name", ""))

        def _gc(key):
            for p in ["config/model/generator/", "config/generator/"]:
                v = row.get(f"{p}{key}")
                if v is not None and str(v) != "nan" and v != "":
                    return v
            return None

        is_bon = "bon" in name.lower()
        sm = _gc("sampling_mode")
        if is_bon or sm:
            sm = sm or "min"
            eta = _gc("eta") or 1.0
            ls = f"$L={int(L)}$" if L and str(L) != "nan" else ""
            if sm == "min":
                return f"PEPO {ls} Rej.Min"
            if sm == "mean_std":
                es = f"{float(eta):.2f}".rstrip("0").rstrip(".")
                return f"PEPO {ls} Rej.\u03b7{es}"
            return f"PEPO {ls} Rej.{sm}"
        gr = _gc("greedy_sampling")
        if gr is True or str(gr).lower() == "true":
            return f"pepo-L{int(L)}" if L and str(L) != "nan" else "pepo"
        return f"pepo-L{int(L)}" if L and str(L) != "nan" else "pepo"

    df["algorithm"] = df.apply(_ss, axis=1)
    if "algorithm" in df.columns and "epoch" in df.columns:
        df = _dedup(df, ["algorithm", "epoch"])
    df = _pepo_wr(df)
    df = _init_wr(df, mid, lc=False)
    if epoch_range is not None and "epoch" in df.columns:
        df = df[(df["epoch"] >= epoch_range[0]) & (df["epoch"] <= epoch_range[1])]
    return df


def find_best_global_L(df, models=None):
    if models is None:
        models = [m for m in MODELS if "Yi" not in m]
    scores = {L: [] for L in [2, 3, 4]}
    for fn, yc in [
        (get_exp1_data, "winrate_initial"),
        (get_mtbench_data, "mtbench_winrate_adjusted"),
    ]:
        for midx, mid in enumerate(MODELS):
            if mid not in models:
                continue
            d = fn(df, midx)
            for L in [2, 3, 4]:
                adf = d[(d["algorithm"] == f"pepo-L{L}") & d[yc].notna()]
                if not adf.empty:
                    scores[L].append(adf.loc[adf["epoch"].idxmax(), yc])
    avgs = {L: (sum(v) / len(v) if v else float("-inf")) for L, v in scores.items()}
    return max(avgs, key=lambda L: (avgs[L], -L))


def _pl(algo):
    m = re.fullmatch(r"pepo-L(\d+)", str(algo))
    return int(m.group(1)) if m else None


def keep_best_pepo_L(df, y_col, hue_col="algorithm", n_last=3, fixed_L=None):
    if (
        df.empty
        or hue_col not in df.columns
        or y_col not in df.columns
        or "epoch" not in df.columns
    ):
        return df
    pd_ = df.copy()
    vl = pd_[hue_col].map(_pl)
    iv = vl.fillna(0) >= 2
    if fixed_L is not None:
        ka = f"pepo-L{fixed_L}"
        cand = pd_.loc[iv, hue_col].unique().tolist()

        def _hd(a):
            adf = pd_[pd_[hue_col] == a].dropna(subset=[y_col])
            return adf["epoch"].max() > 0 if not adf.empty else False

        if ka in cand and _hd(ka):
            ch = ka
        elif cand:

            def _fv(a):
                adf = pd_[pd_[hue_col] == a].dropna(subset=[y_col])
                if adf.empty:
                    return float("-inf")
                return adf.loc[adf["epoch"].idxmax(), y_col]

            ch = max(cand, key=lambda a: (_fv(a), -_pl(a)))
        else:
            return pd_
        pd_ = pd_[~iv | (pd_[hue_col] == ch)].copy()
        pd_.loc[pd_[hue_col] == ch, hue_col] = f"PEPO $L={_pl(ch)}$"
        return pd_

    cand = pd_.loc[iv, hue_col].unique()
    if not len(cand):
        return pd_

    def _mln(a):
        adf = pd_[pd_[hue_col] == a].dropna(subset=[y_col, "epoch"])
        if adf.empty:
            return float("-inf")
        tl = adf[adf["epoch"] > adf["epoch"].max() - n_last][y_col]
        return tl.mean() if len(tl) > 0 else float("-inf")

    ba = max({a: _mln(a) for a in cand}, key=lambda a: (_mln(a), -_pl(a)))
    pd_ = pd_[~iv | (pd_[hue_col] == ba)].copy()
    pd_.loc[pd_[hue_col] == ba, hue_col] = "PEPO"
    return pd_


_ALABEL = {
    "winrate_initial": "Win Rate against the Initial Model (%)",
    "winrate_gpt4": "Win Rate against GPT-4 (%)",
    "winrate_initial_lc": "Length-Controlled Win Rate (%)",
    "mtbench_winrate": "MT-Bench Win Rate (%)",
    "mtbench_winrate_adjusted": "MT-Bench Tie-Adjusted Win Rate (%)",
}


def _lab(s):
    return _ALABEL.get(s, s.replace("_", " ").replace("/", " ").title())


def plot_multi_model_comparison(
    dataframes,
    x_col="epoch",
    y_col="winrate_initial",
    hue_col="algorithm",
    se_col=None,
    exclude_algos=None,
    save_path=None,
    aggregate_best=False,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "figure.dpi": 300,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    am = {
        "pepo-L1": "DPO",
        "dpo": "DPO",
        "sftdpo": r"SFT+DPO",
        "chi2po": r"$\chi^2$PO",
        "pepo-L2": r"PEPO $L=2$",
        "pepo-L3": r"PEPO $L=3$",
        "pepo-L4": r"PEPO $L=4$",
    }
    pds, all_a = [], set()
    for df in dataframes:
        pd_ = df.dropna(subset=[x_col, y_col]).copy()
        if exclude_algos:
            pd_ = pd_[~pd_[hue_col].isin(exclude_algos)]
        pd_["display_algo"] = pd_[hue_col].replace(am)
        if aggregate_best:

            def _mt(a):
                if "Rej." in str(a):
                    return "Rejection Sampling"
                if "pepo-L" in str(a) or "PEPO" in str(a):
                    return r"Token Level"
                return a

            pd_["method"] = pd_["algorithm"].apply(_mt)
            idx = pd_.groupby(["epoch", "method"])[y_col].idxmax()
            pd_ = pd_.loc[idx].copy()
            pd_["display_algo"] = pd_["method"]
        pds.append(pd_)
        all_a.update(pd_["display_algo"].unique())
    ua = sorted(all_a)
    cp = {
        "DPO": "#e74c3c",
        r"SFT+DPO": "#2ecc71",
        r"$\chi^2$PO": "#f1c40f",
        "Rejection Sampling": "#27ae60",
        r"Token Level": "#2980b9",
        "PEPO": "#2980b9",
    }
    bs = sns.color_palette("Blues", 5)[2:]
    gs = sns.color_palette("Greens", 5)[2:]
    ps = sns.color_palette("Purples", 5)[2:]
    for i, a in enumerate(
        sorted(a for a in all_a if "PEPO $L=" in a and "Rej." not in a)
    ):
        cp[a] = bs[i % len(bs)]
    for i, a in enumerate(sorted(a for a in all_a if "Rej.Min" in a)):
        cp[a] = gs[i % len(gs)]
    for i, a in enumerate(sorted(a for a in all_a if "Rej.\u03b7" in a)):
        cp[a] = ps[i % len(ps)]
    cb = sns.color_palette("colorblind", max(len(all_a), 1))
    for i, a in enumerate(ua):
        cp.setdefault(a, cb[i])
    n = len(pds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for idx, (ax, pd_) in enumerate(zip(axes, pds)):
        skw = {"palette": cp} if pd_["display_algo"].nunique() > 1 else {}
        sns.lineplot(
            data=pd_.sort_values(x_col),
            x=x_col,
            y=y_col,
            hue="display_algo",
            **skw,
            linewidth=2,
            ax=ax,
            legend=True,
        )
        if se_col and se_col in pd_.columns:
            for a in pd_["display_algo"].unique():
                adf = pd_[pd_["display_algo"] == a].sort_values(x_col)
                if adf[se_col].isna().all():
                    continue
                ax.fill_between(
                    adf[x_col].values,
                    adf[y_col].values - adf[se_col].fillna(0).values,
                    adf[y_col].values + adf[se_col].fillna(0).values,
                    alpha=0.2,
                    color=cp.get(a, "#888"),
                    linewidth=0,
                )
        if "model" in pd_.columns:
            ax.set_title(pd_["model"].iloc[0].split("/")[-1])
        ax.set_xlabel(_lab(x_col))
        ax.set_ylabel(_lab(y_col) if idx == 0 else "")
        ax.grid(True, alpha=0.2, linestyle="--")
        sns.despine(ax=ax)
        ax.get_legend().remove()
    handles = [plt.Line2D([0], [0], color=cp[a], linewidth=2) for a in ua]
    fig.legend(
        handles,
        ua,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(ua),
        frameon=False,
        fontsize=11,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    if save_path:
        plt.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.05,
            metadata={"Creator": "PEPO Visualization", "Title": "Model Comparison"},
        )
    plt.show()


def plot_main_figure(
    alpaca_best_dfs,
    alpaca_ablation_dfs,
    mtbench_best_dfs,
    mtbench_ablation_dfs,
    save_path=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "figure.dpi": 300,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
        }
    )
    am = {
        "pepo-L1": "DPO",
        "dpo": "DPO",
        "sftdpo": r"SFT+DPO",
        "chi2po": r"$\chi^2$PO",
        "pepo-L2": r"PEPO $L=2$",
        "pepo-L3": r"PEPO $L=3$",
        "pepo-L4": r"PEPO $L=4$",
    }
    rows_cfg = [
        (alpaca_best_dfs, "winrate_initial", "standard_error_initial", []),
        (
            alpaca_ablation_dfs,
            "winrate_initial",
            "standard_error_initial",
            ["sftdpo", "chi2po"],
        ),
        (mtbench_best_dfs, "mtbench_winrate_adjusted", None, []),
        (mtbench_ablation_dfs, "mtbench_winrate_adjusted", None, ["sftdpo", "chi2po"]),
    ]
    panels, all_a = [], set()
    for dfs, y_col, se_col, excl in rows_cfg:
        proc = []
        for df in dfs:
            pd_ = df.dropna(subset=["epoch", y_col]).copy()
            if excl:
                pd_ = pd_[~pd_["algorithm"].isin(excl)]
            pd_["display_algo"] = pd_["algorithm"].replace(am)
            proc.append(pd_)
            all_a.update(pd_["display_algo"].unique())
        panels.append(proc)
    cp = {
        "DPO": "#e74c3c",
        r"SFT+DPO": "#2ecc71",
        r"$\chi^2$PO": "#f1c40f",
        "PEPO": "#2980b9",
    }
    bs = sns.color_palette("Blues", 5)[2:]
    for i, a in enumerate(sorted(a for a in all_a if "PEPO $L=" in a)):
        cp[a] = bs[i % len(bs)]
    rem = [a for a in sorted(all_a) if a not in cp]
    cb = sns.color_palette("colorblind", max(len(rem), 1))
    for i, a in enumerate(rem):
        cp[a] = cb[i]
    ua = sorted(all_a)
    n_mod = len(alpaca_best_dfs)
    fig, axes = plt.subplots(4, n_mod, figsize=(10, 9), sharex="col", sharey="row")
    for ri in range(4):
        proc = panels[ri]
        y_col, se_col = rows_cfg[ri][1], rows_cfg[ri][2]
        for ci in range(n_mod):
            ax = axes[ri, ci]
            df = proc[ci]
            if df.empty:
                ax.set_visible(False)
                continue
            skw2 = {"palette": cp} if df["display_algo"].nunique() > 1 else {}
            sns.lineplot(
                data=df.sort_values("epoch"),
                x="epoch",
                y=y_col,
                hue="display_algo",
                **skw2,
                linewidth=1.8,
                ax=ax,
                legend=False,
            )
            if se_col and se_col in df.columns:
                for a in df["display_algo"].unique():
                    adf = df[df["display_algo"] == a].sort_values("epoch")
                    if adf[se_col].isna().all():
                        continue
                    ax.fill_between(
                        adf["epoch"].values,
                        adf[y_col].values - adf[se_col].fillna(0).values,
                        adf[y_col].values + adf[se_col].fillna(0).values,
                        alpha=0.15,
                        color=cp.get(a, "#888"),
                        linewidth=0,
                    )
            if "model" in df.columns:
                ax.set_title(df["model"].iloc[0].split("/")[-1], fontsize=10)
            ax.set_xlabel("Epoch" if ri == 3 else "")
            ax.set_ylabel("Win Rate (%)" if ci == 0 else "", labelpad=8)
            ax.grid(True, alpha=0.15, linestyle="--")
            sns.despine(ax=ax)
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(bottom=0.09, left=0.07)
    xl = -0.04
    for si, lb in enumerate(["AlpacaEval", "MT-Bench"]):
        top = axes[si * 2, 0].get_position().y1
        bot = axes[si * 2 + 1, 0].get_position().y0
        fig.text(
            xl,
            (top + bot) / 2,
            lb,
            fontsize=13,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
            transform=fig.transFigure,
        )
    order = sorted(a for a in ua if "PEPO $L=" in a) + [
        a for a in ["DPO", r"$\chi^2$PO", r"SFT+DPO"] if a in ua
    ]
    handles = [plt.Line2D([0], [0], color=cp[a], linewidth=2) for a in order]
    fig.legend(
        handles,
        order,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(order),
        frameon=False,
        fontsize=10,
    )
    if save_path:
        plt.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.05,
            metadata={"Creator": "PEPO Visualization", "Title": "Main Figure"},
        )
    plt.show()


def get_best_winrates(
    dataframes,
    y_col="winrate_initial",
    se_col="standard_error_initial",
    hue_col="algorithm",
    exclude_algos=None,
    aggregate_pepo=True,
):
    am = {
        "pepo-L1": "DPO",
        "dpo": "DPO",
        "sftdpo": r"SFT+DPO",
        "chi2po": r"$\chi^2$PO",
        "pepo-L2": "PEPO" if aggregate_pepo else r"PEPO $L=2$",
        "pepo-L3": "PEPO" if aggregate_pepo else r"PEPO $L=3$",
        "pepo-L4": "PEPO" if aggregate_pepo else r"PEPO $L=4$",
    }
    if aggregate_pepo:
        am.update({r"PEPO $L=2$": "PEPO", r"PEPO $L=3$": "PEPO", r"PEPO $L=4$": "PEPO"})
    res = []
    for df in dataframes:
        if df.empty:
            continue
        mn = df["model"].iloc[0].split("/")[-1] if "model" in df.columns else "Unknown"
        pd_ = df.dropna(subset=[y_col]).copy()
        if exclude_algos:
            pd_ = pd_[~pd_[hue_col].isin(exclude_algos)]
        pd_["display_algo"] = pd_[hue_col].replace(am)
        for a in pd_["display_algo"].unique():
            adf = pd_[pd_["display_algo"] == a]
            br = adf.loc[adf[y_col].idxmax()]
            entry = {
                "model": mn,
                "algorithm": a,
                "best_winrate": br[y_col],
                "best_epoch": br.get("epoch"),
            }
            if se_col and se_col in adf.columns:
                entry["standard_error"] = br.get(se_col)
            res.append(entry)
    return pd.DataFrame(res)


def format_winrates_latex(
    summary_df,
    pivot=True,
    bold_best=True,
    include_se=True,
    font_size="small",
    short_names=True,
):
    df = summary_df.copy()
    if short_names:
        df["algorithm"] = df["algorithm"].replace(
            {
                "DPO": "DPO",
                "PEPO": "PEPO",
                r"SFT+DPO": "SFT+DPO",
                r"$\chi^2$PO": r"$\chi^2$PO",
                r"PEPO $L=2$": r"$L{=}2$",
                r"PEPO $L=3$": r"$L{=}3$",
                r"PEPO $L=4$": r"$L{=}4$",
            }
        )
        df["model"] = df["model"].replace(
            {
                "Llama-3.1-Tulu-3-8B-SFT": "Tulu-3-8B",
                "zephyr-7b-sft-full": "Zephyr-7B",
                "mistral-7b-sft-beta": "Mistral-7B",
                "Yi-34B-Chat": "Yi-34B",
            }
        )

    def _fv(r):
        v = f"{r['best_winrate']:.1f}"
        if include_se and "standard_error" in r and pd.notna(r["standard_error"]):
            v += f" $\\pm$ {r['standard_error']:.1f}"
        return v

    df["formatted"] = df.apply(_fv, axis=1)
    if pivot:
        fp = df.pivot(index="model", columns="algorithm", values="formatted").fillna("")
        pv = df.pivot(index="model", columns="algorithm", values="best_winrate")
        pepo_c = [c for c in fp.columns if "PEPO" in c or c == "PEPO"]
        oc = pepo_c + sorted(c for c in fp.columns if c not in pepo_c)
        fp = fp[[c for c in oc if c in fp.columns]]
        pv = pv[[c for c in oc if c in pv.columns]]
        if bold_best:
            for m in pv.index:
                if pv.loc[m].isna().all():
                    continue
                ba = pv.loc[m].idxmax()
                cv = fp.loc[m, ba]
                if cv:
                    fp.loc[m, ba] = f"\\textbf{{{cv}}}"
        fp.index.name = fp.columns.name = None
        latex = fp.to_latex(
            escape=False, column_format="l" + "c" * len(fp.columns), na_rep=""
        )
    else:
        latex = df[["model", "algorithm", "formatted"]].to_latex(
            index=False, escape=False, column_format="llc"
        )
    if font_size and font_size != "normal":
        latex = f"\\{font_size}\n{latex}"
    return latex


def format_exp2_comparison_table(
    df,
    epochs=None,
    y_col="winrate_initial",
    se_col="standard_error_initial",
    L=None,
    aggregate_best=True,
    font_size="footnotesize",
):
    if epochs is None:
        epochs = [1, 2, 3]
    df = df.copy()
    df["method"] = df["algorithm"].apply(
        lambda a: (
            "Rejection Sampling" if "Rej." in str(a) else r"Token Level \texttt{PEPO}"
        )
    )
    if L is not None and "L" in df.columns:
        df = df[df["L"] == L]
    rows = {}
    for m in ["Rejection Sampling", r"Token Level \texttt{PEPO}"]:
        md = df[df["method"] == m]
        rd = {}
        for e in epochs:
            ed = md[md["epoch"] == e]
            if len(ed):
                bi = ed[y_col].idxmax()
                rd[e] = (
                    ed.loc[bi, y_col],
                    ed.loc[bi, se_col] if se_col in ed.columns else None,
                )
            else:
                rd[e] = (None, None)
        rows[m] = rd
    bpe = {}
    for e in epochs:
        bv, bm = -1, None
        for m, rd in rows.items():
            wr = rd[e][0]
            if wr is not None and wr > bv:
                bv, bm = wr, m
        bpe[e] = bm

    def _fc(wr, se, ib):
        if wr is None:
            return ""
        v = f"{wr:.1f}" + (f" $\\pm$ {se:.2f}" if se is not None else "")
        return f"\\textbf{{{v}}}" if ib else v

    lines = [
        r"\begin{table}[h]",
        r"\caption{Win rates of Rejection Sampling "
        r"vs. Token Level \texttt{PEPO}}",
        r"\label{tab:rej-vs-token}",
        r"\centering",
        f"\\{font_size}",
        f"\\begin{{tabular}}{{l{'c' * len(epochs)}}}",
        r"\toprule",
        " & ".join([""] + [f"epoch ${e}$" for e in epochs]) + r" \\",
        r"\midrule",
    ]
    for m in ["Rejection Sampling", r"Token Level \texttt{PEPO}"]:
        cells = [
            _fc(
                rows[m].get(e, (None, None))[0],
                rows[m].get(e, (None, None))[1],
                bpe.get(e) == m,
            )
            for e in epochs
        ]
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def get_final_epoch_winrates(
    dataframes,
    y_col="winrate_initial",
    se_col=None,
    aggregate_pepo=True,
    fixed_L=None,
):
    res = []
    for df in dataframes:
        if df.empty:
            continue
        mn = df["model"].iloc[0].split("/")[-1] if "model" in df.columns else "Unknown"
        pd_ = df.dropna(subset=[y_col]).copy()
        fbs = {}
        for a in pd_["algorithm"].unique():
            adf = pd_[pd_["algorithm"] == a]
            fbs[a] = adf.loc[adf["epoch"].idxmax()]
        am = {
            "pepo-L1": "DPO",
            "dpo": "DPO",
            "sftdpo": r"SFT+DPO",
            "chi2po": r"$\chi^2$PO",
            "pepo-L2": ("PEPO" if aggregate_pepo else r"PEPO $L=2$"),
            "pepo-L3": ("PEPO" if aggregate_pepo else r"PEPO $L=3$"),
            "pepo-L4": ("PEPO" if aggregate_pepo else r"PEPO $L=4$"),
        }
        if fixed_L is not None:
            used = fixed_L
            key = f"pepo-L{fixed_L}"
            if key not in fbs or fbs[key].get("epoch", 0) == 0:
                avail = [
                    L
                    for L in [2, 3, 4]
                    if f"pepo-L{L}" in fbs and fbs[f"pepo-L{L}"].get("epoch", 0) > 0
                ]
                if avail:
                    used = max(
                        avail, key=lambda L: fbs[f"pepo-L{L}"].get(y_col, float("-inf"))
                    )
            for L in [2, 3, 4]:
                if L == used:
                    am[f"pepo-L{L}"] = "PEPO"
                elif f"pepo-L{L}" in am:
                    del am[f"pepo-L{L}"]
        dgs = {}
        for src, row in fbs.items():
            if src in am:
                dgs.setdefault(am[src], []).append(row)
        for dsp, rs in dgs.items():
            best = max(rs, key=lambda r: r[y_col])
            entry = {
                "model": mn,
                "algorithm": dsp,
                "best_winrate": best[y_col],
                "best_epoch": best.get("epoch"),
            }
            if se_col and se_col in best.index:
                entry["standard_error"] = best.get(se_col)
            res.append(entry)
    return pd.DataFrame(res)
