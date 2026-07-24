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


def fetch_runs(entity=None, project="pepo", filters=None):
    """
    Fetches runs from WandB.

    Args:
        entity (str): WandB entity name. Defaults to the WANDB_ENTITY
                      environment variable, falling back to "pepo-team".
        project (str): WandB project name.
        filters (dict, optional): MongoDB-style filter dictionary.

    Returns:
        wandb.Runs: An iterator over run objects.
    """
    entity = entity or os.getenv("WANDB_ENTITY") or "pepo-team"
    api = wandb.Api()
    path = f"{entity}/{project}"
    runs = api.runs(path=path, filters=filters)
    print(f"Found {len(runs)} runs in {path}")
    return runs


def get_runs_df(
    entity=None,
    project="pepo",
    filters=None,
    cache_path=None,
    force_refresh=False,
):
    """
    Fetches runs and returns a DataFrame, with local caching.

    Args:
        entity (str): WandB entity. Defaults to the WANDB_ENTITY
                      environment variable, falling back to "pepo-team".
        project (str): WandB project.
        filters (dict): Filters for WandB.
        cache_path (str): Path to save/load cache. Defaults to .cache/runs_cache.pkl
        force_refresh (bool): If True, ignore cache and re-fetch.

    Returns:
        pd.DataFrame: Runs data.
    """
    entity = entity or os.getenv("WANDB_ENTITY") or "pepo-team"

    if cache_path is None:
        # Default to .cache folder in the same directory as this file
        base_dir = os.path.dirname(__file__)
        cache_dir = os.path.join(base_dir, ".cache")
        cache_path = os.path.join(cache_dir, "runs_cache.csv")

    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    df = None
    if not force_refresh and os.path.exists(cache_path):
        print(f"Loading runs from cache: {cache_path}")
        try:
            # Support pickle if extension is .pkl for better type preservation
            if cache_path.endswith(".pkl"):
                loaded_df = pd.read_pickle(cache_path)
                df = (
                    loaded_df
                    if isinstance(loaded_df, pd.DataFrame)
                    else pd.DataFrame(loaded_df)
                )
            else:
                df = pd.read_csv(cache_path, low_memory=False)

            # Smart refresh logic: use the cache even if W&B is unavailable.
            if "created_at" in df.columns and not df.empty:
                last_time = df["created_at"].max()
                print(
                    f"Cache contains runs up to {last_time}. Checking for new runs..."
                )

                try:
                    # Fetch runs created after the last cached run
                    time_filter = {"created_at": {"$gt": last_time}}
                    if filters:
                        new_filters = {"$and": [filters, time_filter]}
                    else:
                        new_filters = time_filter

                    new_runs = fetch_runs(entity, project, new_filters)
                    new_df = extract_metrics(new_runs)

                    if not new_df.empty:
                        print(f"Found {len(new_df)} new runs. Merging...")
                        df = pd.concat([df, new_df], ignore_index=True)
                        df = df.drop_duplicates(subset=["run_id"], keep="last")
                    else:
                        print("No new runs found.")
                except Exception as e:
                    print(f"W&B incremental refresh skipped, using cache: {e}")
            else:
                print(
                    "Cache missing 'created_at' or empty. "
                    "Forcing full refresh to update schema."
                )
                df = None

        except Exception as e:
            print(f"Error reading cache: {e}. Forcing refresh.")
            df = None

    if df is None:
        print("Fetching all runs from WandB...")
        try:
            runs = fetch_runs(entity, project, filters)
            df = extract_metrics(runs)
        except Exception as e:
            if not force_refresh and os.path.exists(cache_path):
                print(f"W&B full fetch failed ({e}). Using existing cache.")
                if cache_path.endswith(".pkl"):
                    loaded_df = pd.read_pickle(cache_path)
                    df = (
                        loaded_df
                        if isinstance(loaded_df, pd.DataFrame)
                        else pd.DataFrame(loaded_df)
                    )
                else:
                    df = pd.read_csv(cache_path, low_memory=False)
            else:
                raise

    print(f"Saving cache to {cache_path}")
    if cache_path.endswith(".pkl"):
        df.to_pickle(cache_path)
    else:
        df.to_csv(cache_path, index=False)

    return df


def flatten_dict(d, parent_key="", sep="/"):
    """
    Recursively flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def inspect_run(run):
    """
    Prints the structure of a run's config and summary.
    """
    print(f"--- Run: {run.name} ({run.id}) ---")
    print("Config Keys:")
    flat_config = flatten_dict(run.config, parent_key="config")
    for k in sorted(flat_config.keys()):
        print(f"  {k}")

    print("\nSummary Keys:")
    flat_summary = flatten_dict(run.summary, parent_key="summary")
    for k in sorted(flat_summary.keys()):
        print(f"  {k}")


def extract_metrics(runs):
    """
    Extracts config and summary metrics from runs into a DataFrame.

    Args:
        runs: Iterator of WandB runs.

    Returns:
        pd.DataFrame: DataFrame containing run data.
    """
    data = []
    # If runs is an iterator, we need to be careful not to exhaust it
    # if we want to use it again.
    # But usually one passes a list or consumes it.

    def serialize_value(v):
        if isinstance(v, (dict, list, tuple)):
            return str(v)
        return v

    for run in runs:
        # Use our custom flattener
        flat_config = flatten_dict(run.config, parent_key="config")
        flat_summary = flatten_dict(run.summary, parent_key="summary")

        entry = {
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            **{k: serialize_value(v) for k, v in flat_config.items()},
            **{k: serialize_value(v) for k, v in flat_summary.items()},
        }
        data.append(entry)

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def process_history(run, keys=None, samples=0):
    """
    Fetches history for a specific run.

    Args:
        run: WandB run object.
        keys (list): List of metric keys to fetch.
        samples (int): Number of history samples to fetch. 0 for all.

    Returns:
        pd.DataFrame: History dataframe.
    """
    # use scan_history for efficiency if grabbing all
    if samples == 0:
        # scan_history yields dicts
        history = pd.DataFrame(run.scan_history(keys=keys))
    else:
        if keys is None:
            history = run.history(samples=samples)
        else:
            history = run.history(keys=keys, samples=samples)
    return history


def extract_history(runs, keys=None, samples=500):
    """
    Fetches history for multiple runs and concatenates them.

    Args:
        runs: Iterable of WandB runs.
        keys (list): Specific metrics to fetch (e.g., ['train/loss']).
                     If None, fetches everything (can be slow).
        samples (int): Number of samples per run.

    Returns:
        pd.DataFrame: Long-format DataFrame with run_id and name.
    """
    all_history = []
    for run in runs:
        try:
            hist = process_history(run, keys=keys, samples=samples)
            if not hist.empty:
                hist["run_id"] = run.id
                hist["name"] = run.name
                # Add config columns if needed? For now just ID.
                all_history.append(hist)
        except Exception as e:
            print(f"Failed to fetch history for run {run.name}: {e}")

    if not all_history:
        return pd.DataFrame()

    return pd.concat(all_history, ignore_index=True)


def _is_alpaca_eval_run(names):
    """True for AlpacaEval runs; excludes MT-Bench eval runs."""
    is_eval = names.str.contains("eval", na=False)
    is_mtbench = names.str.contains("mtbench", na=False)
    return is_eval & ~is_mtbench


def deduplicate_runs(df, group_cols, strategy="last", time_col="created_at"):
    """
    When multiple runs share the same group keys (e.g. same algorithm
    and epoch), keeps one representative according to *strategy*.

    Strategies
    ----------
    ``"last"``
        Keep the run with the latest *time_col* value.
    ``"best"``
        Keep the run with the highest value in the group's *y_col*.
    ``"mean"``
        Average all numeric columns within the group (placeholder).
    """
    if df.empty or not all(c in df.columns for c in group_cols):
        return df

    # Defragment to avoid PerformanceWarning from groupby on a highly-fragmented df
    df = df.copy()

    if strategy == "last":
        if time_col not in df.columns:
            return df
        return df.sort_values(time_col).groupby(group_cols, as_index=False).last()
    elif strategy == "best":
        raise NotImplementedError("best strategy requires a y_col parameter")
    elif strategy == "mean":
        raise NotImplementedError("mean strategy not yet implemented")
    else:
        raise ValueError(f"Unknown deduplication strategy: {strategy}")


def get_exp1_data(df, model_idx=0):
    """
    Processes dataframe for Experiment 1.
    """
    model_id = MODELS[model_idx]
    # Filter by model_id in various config locations
    df = df[
        (df["config/model/model_id"] == model_id)
        | (df["config/backbone/model_id"] == model_id)
        | (df["config/model/backbone/model_id"] == model_id)
    ]

    # Filter for greedy sampling
    if "config/model/generator/greedy_sampling" in df.columns:
        greedy_sampling = df["config/model/generator/greedy_sampling"].map(
            lambda value: value is True or str(value).lower() == "true"
        )
        df = df[greedy_sampling].copy()

    # Filter for AlpacaEval runs (exclude MT-Bench eval runs)
    df = df[_is_alpaca_eval_run(df["name"])]

    # Create derived columns
    if "config/L" in df.columns:
        df["L"] = df["config/L"]

    if "summary/eval/epoch" in df.columns:
        df["epoch"] = df["summary/eval/epoch"]

    df["model"] = model_id

    # Handle tags parsing
    if "config/model/wandb/tags" in df.columns:
        df["config/model/wandb/tags"] = df["config/model/wandb/tags"].apply(
            lambda x: (
                ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
            )
        )

        def get_algorithm(row):
            tags = row["config/model/wandb/tags"]
            # If it is a list object and not empty
            if isinstance(tags, list) and len(tags) > 0:
                return tags[0]
            # Fallback
            return f"pepo-L{row['L']}" if "L" in row else "unknown"

        df["algorithm"] = df.apply(get_algorithm, axis=1)

    # Deduplicate: keep the last run per (algorithm, epoch)
    if "algorithm" in df.columns and "epoch" in df.columns:
        df = deduplicate_runs(df, group_cols=["algorithm", "epoch"])

    # Extract winrates
    gpt4_cols = "summary/eval/tatsu-lab/alpaca_eval/mt1024/"
    win_col = f"{gpt4_cols}win_rate"
    se_col = f"{gpt4_cols}standard_error"
    if win_col in df.columns:
        df["winrate_gpt4"] = df[win_col]
    if se_col in df.columns:
        df["standard_error_gpt4"] = df[se_col]

    model_suffix = model_id.split("/")[-1]
    initial_cols = (
        f"summary/eval/tatsu-lab/alpaca_eval/mt1024/{model_suffix}-a0.0-b0.1-L1/"
    )
    win_col = f"{initial_cols}win_rate"
    se_col = f"{initial_cols}standard_error"

    if win_col in df.columns:
        df["winrate_initial"] = df[win_col]
        # fill NaN in initial with 50 if epoch is 0
        if "epoch" in df.columns:
            df.loc[df["epoch"] == 0, "winrate_initial"] = 50.0

    if se_col in df.columns:
        df["standard_error_initial"] = df[se_col]

    # Length-controlled variants (correct for verbosity bias)
    lc_win_col = f"{initial_cols}length_controlled_winrate"
    lc_se_col = f"{initial_cols}lc_standard_error"

    if lc_win_col in df.columns:
        df["winrate_initial_lc"] = df[lc_win_col]
        if "epoch" in df.columns:
            df.loc[df["epoch"] == 0, "winrate_initial_lc"] = 50.0

    if lc_se_col in df.columns:
        df["standard_error_initial_lc"] = df[lc_se_col]

    # Ensure epoch 0 exists for every algorithm
    if (
        "algorithm" in df.columns
        and "epoch" in df.columns
        and "winrate_initial" in df.columns
    ):
        new_rows = []
        for algo in df["algorithm"].unique():
            algo_df = df[df["algorithm"] == algo]
            # Check if 0 is present in the epoch column
            # (ignoring float/int differences close to 0)
            if not ((algo_df["epoch"] == 0).any()):
                # Create a synthetic row based on existing data to verify structure
                base_row = algo_df.iloc[0].copy()
                base_row["epoch"] = 0
                base_row["winrate_initial"] = 50.0
                if "winrate_initial_lc" in df.columns:
                    base_row["winrate_initial_lc"] = 50.0
                # Clear run specific info that might be confusing if duplicated?
                # For plotting purposes, copying is fine.
                new_rows.append(base_row)

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df


def _parse_wandb_tags(value):
    if isinstance(value, str) and value.startswith("["):
        return ast.literal_eval(value)
    return value


def _algorithm_from_tags_or_l(row):
    tags = row.get("config/model/wandb/tags")
    if isinstance(tags, list) and len(tags) > 0:
        return tags[0]

    L = row.get("L", row.get("config/L", None))
    if pd.notna(L):
        return f"pepo-L{int(L)}"

    return "unknown"


def get_mtbench_data(df, model_idx=0):
    """
    Processes dataframe for MT-Bench win-rate plots.
    """
    model_id = MODELS[model_idx]
    df = df[
        (df["config/model/model_id"] == model_id)
        | (df["config/backbone/model_id"] == model_id)
        | (df["config/model/backbone/model_id"] == model_id)
    ].copy()

    # MT-Bench eval runs are named with mtbench-eval and carry mt_bench metrics.
    df = df[df["name"].str.contains("mtbench-eval", na=False)].copy()

    metric_prefix = "summary/eval/mt_bench/mt1024/"
    win_col = f"{metric_prefix}win_rate"
    adjusted_win_col = f"{metric_prefix}win_rate_adjusted"
    if win_col in df.columns:
        df["mtbench_winrate"] = df[win_col] * 100.0

    if adjusted_win_col in df.columns:
        df["mtbench_winrate_adjusted"] = df[adjusted_win_col] * 100.0

    score_col = f"{metric_prefix}score"
    if score_col in df.columns:
        df["mtbench_score"] = df[score_col]

    if "summary/eval/epoch" in df.columns:
        df["epoch"] = df["summary/eval/epoch"]

    if "config/L" in df.columns:
        df["L"] = df["config/L"]

    df["model"] = model_id

    if "config/model/wandb/tags" in df.columns:
        df["config/model/wandb/tags"] = df["config/model/wandb/tags"].apply(
            _parse_wandb_tags
        )

    df["algorithm"] = df.apply(_algorithm_from_tags_or_l, axis=1)

    # Deduplicate: keep the last run per (algorithm, epoch)
    if "algorithm" in df.columns and "epoch" in df.columns:
        df = deduplicate_runs(df, group_cols=["algorithm", "epoch"])

    if "epoch" in df.columns and "mtbench_winrate_adjusted" in df.columns:
        df.loc[df["epoch"] == 0, "mtbench_winrate_adjusted"] = 50.0

    if (
        "algorithm" in df.columns
        and "epoch" in df.columns
        and "mtbench_winrate_adjusted" in df.columns
    ):
        new_rows = []
        for algo in df["algorithm"].dropna().unique():
            algo_df = df[df["algorithm"] == algo]
            if algo_df.empty or (algo_df["epoch"] == 0).any():
                continue

            base_row = algo_df.iloc[0].copy()
            base_row["epoch"] = 0
            base_row["mtbench_winrate_adjusted"] = 50.0
            new_rows.append(base_row)

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df


def get_exp2_data(df, model_idx=0, epoch_range=None):
    """
    Processes dataframe for Experiment 2: comparing PEPO sampling strategies.

    Args:
        df: Raw runs dataframe.
        model_idx: Index into MODELS list.
        epoch_range: Optional tuple (min_epoch, max_epoch) to filter epochs.

    Compares:
    - PEPO-G: Greedy (token-level) sampling
    - PEPO-BON-MIN: Best-of-N with min mode
    - PEPO-BON-η{value}: Best-of-N with mean_std mode, different eta values
    """
    model_id = MODELS[model_idx]

    # Filter by model_id in various config locations
    df = df[
        (df["config/model/model_id"] == model_id)
        | (df["config/backbone/model_id"] == model_id)
        | (df["config/model/backbone/model_id"] == model_id)
    ].copy()

    # Filter for AlpacaEval runs only (exclude MT-Bench eval runs)
    df = df[_is_alpaca_eval_run(df["name"])]

    # Identify BON runs by name pattern (they should be kept even without L >= 2)
    def is_bon_run(row):
        name = str(row.get("name", ""))
        return "bon" in name.lower()

    # Filter: keep PEPO runs (L >= 2) OR BON runs
    if "config/L" in df.columns:
        is_pepo = df["config/L"] >= 2
        is_bon = df.apply(is_bon_run, axis=1)
        df = df[is_pepo | is_bon].copy()
        df["L"] = df["config/L"]
    else:
        df = df.copy()

    if "summary/eval/epoch" in df.columns:
        df["epoch"] = df["summary/eval/epoch"]

    df["model"] = model_id

    # Determine sampling strategy from generator config
    def get_sampling_strategy(row):
        L = row.get("L", row.get("config/L", ""))
        # L_suffix was unused

        name = str(row.get("name", ""))

        # Check multiple possible paths for generator config
        prefixes = ["config/model/generator/", "config/generator/"]

        def get_config(key):
            for prefix in prefixes:
                val = row.get(f"{prefix}{key}")
                if val is not None and val != "" and str(val) != "nan":
                    return val
            return None

        # Check if this is a BON run (by name pattern or sampling_mode presence)
        is_bon = "bon" in name.lower()
        sampling_mode = get_config("sampling_mode")

        if is_bon or sampling_mode:
            # This is a BON (rejection sampling) run
            sampling_mode = sampling_mode or "min"
            eta = get_config("eta") or 1.0
            L = row.get("L", row.get("config/L", ""))
            L_str = f"$L={int(L)}$" if L and str(L) != "nan" else ""

            if sampling_mode == "min":
                return f"PEPO {L_str} Rej.Min"
            elif sampling_mode == "mean_std":
                eta_str = f"{float(eta):.2f}".rstrip("0").rstrip(".")
                return f"PEPO {L_str} Rej.η{eta_str}"
            else:
                return f"PEPO {L_str} Rej.{sampling_mode}"

        # Not BON - this is greedy/token-level sampling (same as exp1)
        greedy = get_config("greedy_sampling")
        if greedy is True or str(greedy).lower() == "true":
            # Use same naming as exp1 for consistency
            L = row.get("L", row.get("config/L", ""))
            return f"pepo-L{int(L)}" if L and str(L) != "nan" else "pepo"

        # Fallback: assume greedy (token-level)
        L = row.get("L", row.get("config/L", ""))
        return f"pepo-L{int(L)}" if L and str(L) != "nan" else "pepo"

    df["algorithm"] = df.apply(get_sampling_strategy, axis=1)

    # Deduplicate: keep the last run per (algorithm, epoch)
    if "algorithm" in df.columns and "epoch" in df.columns:
        df = deduplicate_runs(df, group_cols=["algorithm", "epoch"])

    # Extract winrates (same as exp1)
    gpt4_cols = "summary/eval/tatsu-lab/alpaca_eval/mt1024/"
    win_col = f"{gpt4_cols}win_rate"
    se_col = f"{gpt4_cols}standard_error"
    if win_col in df.columns:
        df["winrate_gpt4"] = df[win_col]
    if se_col in df.columns:
        df["standard_error_gpt4"] = df[se_col]

    model_suffix = model_id.split("/")[-1]
    initial_cols = (
        f"summary/eval/tatsu-lab/alpaca_eval/mt1024/{model_suffix}-a0.0-b0.1-L1/"
    )
    win_col = f"{initial_cols}win_rate"
    se_col = f"{initial_cols}standard_error"

    if win_col in df.columns:
        df["winrate_initial"] = df[win_col]
        if "epoch" in df.columns:
            df.loc[df["epoch"] == 0, "winrate_initial"] = 50.0

    if se_col in df.columns:
        df["standard_error_initial"] = df[se_col]

    # Filter by epoch range if specified
    if epoch_range is not None and "epoch" in df.columns:
        min_epoch, max_epoch = epoch_range
        df = df[(df["epoch"] >= min_epoch) & (df["epoch"] <= max_epoch)]

    return df


def find_best_global_L(df, models=None):
    """Find the single L (2, 3, or 4) with the highest avg final-epoch win rate
    across models and benchmarks."""
    if models is None:
        models = [m for m in MODELS if "Yi" not in m]
    scores = {L: [] for L in [2, 3, 4]}
    for get_data, y_col in [
        (get_exp1_data, "winrate_initial"),
        (get_mtbench_data, "mtbench_winrate_adjusted"),
    ]:
        for midx, model_id in enumerate(MODELS):
            if model_id not in models:
                continue
            d = get_data(df, midx)
            for L in [2, 3, 4]:
                adf = d[(d["algorithm"] == f"pepo-L{L}") & d[y_col].notna()]
                if not adf.empty:
                    last = adf.loc[adf["epoch"].idxmax()]
                    scores[L].append(last[y_col])
    avgs = {L: (sum(v) / len(v) if v else float("-inf")) for L, v in scores.items()}
    return max(avgs, key=lambda L: (avgs[L], -L))


def keep_best_pepo_L(df, y_col, hue_col="algorithm", n_last=3, fixed_L=None):
    """
    Reduces PEPO L variants (pepo-L2, pepo-L3, ...) to the single best L,
    renamed to "PEPO" for display. All other algorithms are kept unchanged.

    The best L is the one with the highest average *y_col* over its last
    *n_last* epochs (i.e., the converged performance).

    Args:
        df (pd.DataFrame): DataFrame from get_exp1_data()/get_mtbench_data().
        y_col (str): Metric column used to rank the L variants.
        hue_col (str): Column containing algorithm names.
        n_last (int): Number of final epochs to average over.

    Returns:
        pd.DataFrame: Copy of df with only the best PEPO L variant kept.
    """
    if (
        df.empty
        or hue_col not in df.columns
        or y_col not in df.columns
        or "epoch" not in df.columns
    ):
        return df

    def parse_l(algo):
        match = re.fullmatch(r"pepo-L(\d+)", str(algo))
        return int(match.group(1)) if match else None

    plot_df = df.copy()
    variant_l = plot_df[hue_col].map(parse_l)
    is_variant = variant_l.fillna(0) >= 2

    if fixed_L is not None:
        keep_algo = f"pepo-L{fixed_L}"
        candidates = plot_df.loc[is_variant, hue_col].unique().tolist()

        def has_data(algo):
            adf = plot_df[plot_df[hue_col] == algo].dropna(subset=[y_col])
            return adf["epoch"].max() > 0 if not adf.empty else False

        if keep_algo in candidates and has_data(keep_algo):
            chosen = keep_algo
        elif candidates:
            # Fallback to the variant with the highest final-epoch value
            def final_val(algo):
                adf = plot_df[plot_df[hue_col] == algo].dropna(subset=[y_col])
                if adf.empty:
                    return float("-inf")
                return adf.loc[adf["epoch"].idxmax(), y_col]

            chosen = max(candidates, key=lambda a: (final_val(a), -parse_l(a)))
        else:
            return plot_df
        plot_df = plot_df[~is_variant | (plot_df[hue_col] == chosen)].copy()
        L_used = parse_l(chosen)
        plot_df.loc[plot_df[hue_col] == chosen, hue_col] = f"PEPO $L={L_used}$"
        return plot_df

    candidates = plot_df.loc[is_variant, hue_col].unique()

    if len(candidates) == 0:
        return plot_df

    def mean_last_n(algo):
        adf = plot_df[plot_df[hue_col] == algo].dropna(subset=[y_col, "epoch"])
        if adf.empty:
            return float("-inf")
        last_epoch = adf["epoch"].max()
        tail = adf[adf["epoch"] > last_epoch - n_last][y_col]
        return tail.mean() if len(tail) > 0 else float("-inf")

    scores = {algo: mean_last_n(algo) for algo in candidates}
    best_algo = max(scores, key=lambda a: (scores[a], -parse_l(a)))

    plot_df = plot_df[~is_variant | (plot_df[hue_col] == best_algo)].copy()
    plot_df.loc[plot_df[hue_col] == best_algo, hue_col] = "PEPO"
    return plot_df


def plot_metrics_over_epochs(
    df,
    x_col="epoch",
    y_col="winrate_initial",
    hue_col="algorithm",
    title=None,
    exclude_algos=None,
):
    """
    Plots metrics over epochs with improved styling and custom colors.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        x_col (str): Column name for x-axis (default: 'epoch').
        y_col (str): Column name for y-axis (default: 'winrate_initial').
        hue_col (str): Column name for grouping lines (default: 'algorithm').
        title (str, optional): Title for the plot.
        exclude_algos (list, optional): List of algorithms (original names)
                                        to exclude from the plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use serif fonts
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "figure.dpi": 300,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # Filter out rows where x or y are NaN
    plot_df = df.dropna(subset=[x_col, y_col]).sort_values(x_col).copy()

    # Filter out excluded algorithms
    if exclude_algos:
        plot_df = plot_df[~plot_df[hue_col].isin(exclude_algos)]

    # 1. Rename algorithms and map labels
    # Tag pepo-L1 as DPO earlier or handle it in the mapping
    # Mapping for display names
    algo_map = {
        "pepo-L1": "DPO",
        "dpo": "DPO",
        "sftdpo": r"SFT+DPO",
        "chi2po": r"$\chi^2$PO",
        "pepo-L2": r"PEPO $L=2$",
        "pepo-L3": r"PEPO $L=3$",
        "pepo-L4": r"PEPO $L=4$",
    }

    # Apply mapping to dataframe column used for hue
    # Use a new column to allow flexibility if needed
    plot_df["display_algo"] = plot_df[hue_col].replace(algo_map)

    # Ensure all algorithms are present in the palette, even if not in map
    # PEPO variants get shades of blue/purple, others get distinct colors

    unique_algos = sorted(plot_df["display_algo"].unique())

    # Define base custom palette
    # Adjust these hex codes as needed for "better" colors
    custom_palette = {}

    # PEPO shades (e.g., Blues or Purples)
    # L=2, L=3, L=4
    pepo_shades = sns.color_palette("Blues", n_colors=5)[2:]  # Start from darker shades

    # Assign specific colors
    custom_palette["DPO"] = "#e74c3c"  # Red
    custom_palette[r"SFT+DPO"] = "#2ecc71"  # Green
    custom_palette[r"$\chi^2$PO"] = "#f1c40f"  # Yellow/Orange or distinct

    # Map PEPO levels to shades
    pepo_levels = [a for a in unique_algos if "PEPO" in a]
    for i, algo in enumerate(pepo_levels):
        # Cyclical or distributed assignment if more levels than expected
        custom_palette[algo] = pepo_shades[i % len(pepo_shades)]

    # Fill in any missing keys with default palette colors
    default_colors = sns.color_palette("colorblind", n_colors=len(unique_algos))
    for i, algo in enumerate(unique_algos):
        if algo not in custom_palette:
            custom_palette[algo] = default_colors[i]

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue="display_algo",
        palette=custom_palette,
        linewidth=2,
    )

    # Remove top and right spines
    sns.despine()

    # Helper to clean labels
    def format_label(s):
        if s == "winrate_initial":
            return "Win Rate against the Initial Model (%)"
        if s == "winrate_initial_lc":
            return "Length-Controlled Win Rate (%)"
        if s == "winrate_gpt4":
            return "Win Rate against GPT-4 (%)"
        if s == "mtbench_winrate":
            return "MT-Bench Win Rate (%)"
        if s == "mtbench_winrate_adjusted":
            return "MT-Bench Tie-Adjusted Win Rate (%)"
        if s == "mtbench_score":
            return "MT-Bench Score"
        return s.replace("_", " ").replace("/", " ").title()

    if title:
        plt.title(title)
    else:
        # Try to get model_id if available
        # (assuming it's constant for the plot or picking the first one)
        model_part = ""
        if "model" in plot_df.columns:
            # Just take the base name of the first model
            model_name = plot_df["model"].iloc[0].split("/")[-1]
            model_part = f" for {model_name}"

        plt.title(f"{format_label(y_col)} vs {format_label(x_col)}{model_part}")

    plt.xlabel(format_label(x_col))
    plt.ylabel(format_label(y_col))

    plt.grid(True, alpha=0.2, linestyle="--")

    # Move legend to bottom right and remove frame
    # Sort legend items? usually seaborn handles it well based on hue order
    plt.legend(loc="lower right", frameon=False, title=None)

    plt.tight_layout()
    plt.show()


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
    """
    Creates a 1×N subplot figure comparing metrics across multiple models.

    Args:
        dataframes (list): List of DataFrames, one per model (from get_exp1_data).
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        hue_col (str): Column name for grouping lines.
        se_col (str, optional): Column name for standard error
                                (e.g., 'standard_error_initial').
                                If provided, shaded error regions are drawn
                                around lines.
        exclude_algos (list, optional): Algorithms to exclude.
        save_path (str, optional): Path to save the figure (e.g., 'figure.pdf').
        aggregate_best (bool): If True, aggregates Rejection Sampling vs
                               Token Level PEPO and plots best.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_models = len(dataframes)

    # Use serif fonts
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

    # Mapping for display names (supports both exp1 and exp2 algorithms)
    algo_map = {
        # Exp1 algorithms
        "pepo-L1": "DPO",
        "dpo": "DPO",
        "sftdpo": r"SFT+DPO",
        "chi2po": r"$\chi^2$PO",
        "pepo-L2": r"PEPO $L=2$",
        "pepo-L3": r"PEPO $L=3$",
        "pepo-L4": r"PEPO $L=4$",
        # Exp2 algorithms - keep as-is for now
    }

    # Pre-process dataframes if aggregation is requested
    processed_dfs = []

    # Helper to determine method type
    def get_method_type(algo):
        if "Rej." in str(algo):
            return "Rejection Sampling"
        elif "pepo-L" in str(algo) or "PEPO" in str(algo):
            # Exclude base DPO/SFT+DPO from aggregation if they are
            # present in algos list?
            # Assuming we only aggregate PEPO variants
            return r"Token Level"
        return algo  # Keep others as is

    for df in dataframes:
        plot_df = df.dropna(subset=[x_col, y_col]).copy()
        if exclude_algos:
            plot_df = plot_df[~plot_df[hue_col].isin(exclude_algos)]

        plot_df["display_algo"] = plot_df[hue_col].replace(algo_map)

        if aggregate_best:
            # Group by epoch and method type, take max winrate
            plot_df["method"] = plot_df["algorithm"].apply(get_method_type)

            # Keep only the rows that correspond to the max winrate per (epoch, method)
            # This preserves the correct standard error from the best run
            idx = plot_df.groupby(["epoch", "method"])[y_col].idxmax()
            plot_df = plot_df.loc[idx].copy()
            plot_df["display_algo"] = plot_df["method"]

        processed_dfs.append(plot_df)

    # Collect all unique algorithms across all processed dataframes
    # for consistent coloring
    all_algos = set()
    for df in processed_dfs:
        all_algos.update(df["display_algo"].unique())

    unique_algos = sorted(all_algos)

    # Build color palette with different base colors for different types
    custom_palette = {}

    # Define base color palettes for each type

    bon_min_shades = sns.color_palette("Greens", n_colors=5)[2:]  # Greens for BON-MIN
    bon_eta_shades = sns.color_palette("Purples", n_colors=5)[2:]  # Purples for BON-η
    pepo_shades = sns.color_palette("Blues", n_colors=5)[
        2:
    ]  # Blues for PEPO L variants

    # Baselines
    custom_palette["DPO"] = "#e74c3c"
    custom_palette[r"SFT+DPO"] = "#2ecc71"
    custom_palette[r"$\chi^2$PO"] = "#f1c40f"

    # Aggregated special colors
    custom_palette["Rejection Sampling"] = "#27ae60"  # Strong Green
    custom_palette[r"Token Level \texttt{PEPO}"] = "#2980b9"  # Strong Blue
    custom_palette["PEPO"] = "#2980b9"  # Strong Blue (best-L PEPO)

    # Assign colors by type (for non-aggregated or remaining items)
    rej_min_algos = [a for a in unique_algos if "Rej.Min" in a]
    rej_eta_algos = [a for a in unique_algos if "Rej.η" in a]
    pepo_L_algos = [a for a in unique_algos if "PEPO $L=" in a and "Rej." not in a]

    for i, algo in enumerate(rej_min_algos):
        custom_palette[algo] = bon_min_shades[i % len(bon_min_shades)]  # Green
    for i, algo in enumerate(rej_eta_algos):
        custom_palette[algo] = bon_eta_shades[i % len(bon_eta_shades)]  # Purple
    for i, algo in enumerate(pepo_L_algos):
        custom_palette[algo] = pepo_shades[i % len(pepo_shades)]  # Blue

    # Fill remaining with default colors
    default_colors = sns.color_palette("colorblind", n_colors=len(unique_algos))
    for i, algo in enumerate(unique_algos):
        if algo not in custom_palette:
            custom_palette[algo] = default_colors[i]

    # Create figure with 1 row, n_models columns
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)

    # Ensure axes is iterable even if n_models == 1
    if n_models == 1:
        axes = [axes]

    def format_label(s):
        if s == "winrate_initial":
            return "Win Rate against the Initial Model (%)"
        if s == "winrate_initial_lc":
            return "Length-Controlled Win Rate (%)"
        if s == "winrate_gpt4":
            return "Win Rate against GPT-4 (%)"
        if s == "mtbench_winrate":
            return "MT-Bench Win Rate (%)"
        if s == "mtbench_winrate_adjusted":
            return "MT-Bench Tie-Adjusted Win Rate (%)"
        if s == "mtbench_score":
            return "MT-Bench Score"
        return s.replace("_", " ").replace("/", " ").title()

    for idx, (ax, df) in enumerate(zip(axes, processed_dfs)):
        plot_df = df.sort_values(x_col).copy()

        # Filtering and algo mapping already done in pre-processing step

        sns.lineplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue="display_algo",
            palette=custom_palette,
            linewidth=2,
            ax=ax,
            legend=True,  # Enable legend on all to collect handles
        )

        # Draw error bands if se_col is provided
        if se_col and se_col in plot_df.columns:
            for algo in plot_df["display_algo"].unique():
                algo_data = plot_df[plot_df["display_algo"] == algo].sort_values(x_col)
                # Skip if no valid standard error data
                if algo_data[se_col].isna().all():
                    continue
                x_vals = algo_data[x_col].values
                y_vals = algo_data[y_col].values
                se_vals = algo_data[se_col].fillna(0).values
                ax.fill_between(
                    x_vals,
                    y_vals - se_vals,
                    y_vals + se_vals,
                    alpha=0.2,
                    color=custom_palette.get(algo, "#888888"),
                    linewidth=0,
                )

        # Title: model name
        if "model" in plot_df.columns:
            model_name = plot_df["model"].iloc[0].split("/")[-1]
            ax.set_title(model_name)

        # X label for all
        ax.set_xlabel(format_label(x_col))

        # Y label only for first subplot
        if idx == 0:
            ax.set_ylabel(format_label(y_col))
        else:
            ax.set_ylabel("")

        ax.grid(True, alpha=0.2, linestyle="--")
        sns.despine(ax=ax)

        # Remove individual legends, we'll create a unified one
        ax.get_legend().remove()

    # Collect all unique handles and labels for the unified legend
    # Create a dummy plot to get consistent legend entries for all algorithms
    handles = []
    labels = []
    for algo in unique_algos:
        line = plt.Line2D([0], [0], color=custom_palette[algo], linewidth=2)
        handles.append(line)
        labels.append(algo)

    # Create unified legend below the plots
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(unique_algos),  # All items in one row
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout()

    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.15)

    if save_path:
        # Use 'pdf' backend for true vector output (smallest size, infinite quality)
        # bbox_inches='tight' removes excess whitespace
        # pad_inches controls margin around figure
        plt.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.05,
            # For PDF, dpi only affects rasterized elements (none in line plots)
            # metadata can be added for better document properties
            metadata={"Creator": "PEPO Visualization", "Title": "Model Comparison"},
        )
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_main_figure(
    alpaca_best_dfs,
    alpaca_ablation_dfs,
    mtbench_best_dfs,
    mtbench_ablation_dfs,
    save_path=None,
):
    """
    Creates a 4×3 figure (4 rows × 3 models) combining AlpacaEval and
    MT-Bench methods & L-ablation panels, with a single shared legend.

    Section labels appear as column-spanning headings on the left of
    each pair of rows.
    """
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

    n_models = 3
    row_configs = [
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

    algo_map = {
        "pepo-L1": "DPO",
        "dpo": "DPO",
        "sftdpo": r"SFT+DPO",
        "chi2po": r"$\chi^2$PO",
        "pepo-L2": r"PEPO $L=2$",
        "pepo-L3": r"PEPO $L=3$",
        "pepo-L4": r"PEPO $L=4$",
    }

    panels = []
    all_algos = set()
    for dfs, y_col, se_col, exclude in row_configs:
        processed = []
        for df in dfs:
            plot_df = df.dropna(subset=["epoch", y_col]).copy()
            if exclude:
                plot_df = plot_df[~plot_df["algorithm"].isin(exclude)]
            plot_df["display_algo"] = plot_df["algorithm"].replace(algo_map)
            processed.append(plot_df)
            all_algos.update(plot_df["display_algo"].unique())
        panels.append(processed)

    custom_palette = {
        "DPO": "#e74c3c",
        r"SFT+DPO": "#2ecc71",
        r"$\chi^2$PO": "#f1c40f",
        "PEPO": "#2980b9",
    }
    pepo_shades = sns.color_palette("Blues", n_colors=5)[2:]
    for i, algo in enumerate(sorted(a for a in all_algos if "PEPO $L=" in a)):
        custom_palette[algo] = pepo_shades[i % len(pepo_shades)]
    remaining = [a for a in sorted(all_algos) if a not in custom_palette]
    cb_palette = sns.color_palette("colorblind", n_colors=len(remaining))
    for i, algo in enumerate(remaining):
        custom_palette[algo] = cb_palette[i]
    unique_algos = sorted(all_algos)

    fig, axes = plt.subplots(4, n_models, figsize=(10, 9), sharex="col", sharey="row")

    for row_idx in range(4):
        processed = panels[row_idx]
        y_col = row_configs[row_idx][1]
        se_col = row_configs[row_idx][2]

        for col_idx in range(n_models):
            ax = axes[row_idx, col_idx]
            df = processed[col_idx]

            if df.empty:
                ax.set_visible(False)
                continue

            plot_data = df.sort_values("epoch")
            sns.lineplot(
                data=plot_data,
                x="epoch",
                y=y_col,
                hue="display_algo",
                palette=custom_palette,
                linewidth=1.8,
                ax=ax,
                legend=False,
            )

            if se_col and se_col in df.columns:
                for algo in df["display_algo"].unique():
                    adf = df[df["display_algo"] == algo].sort_values("epoch")
                    if adf[se_col].isna().all():
                        continue
                    ax.fill_between(
                        adf["epoch"].values,
                        adf[y_col].values - adf[se_col].fillna(0).values,
                        adf[y_col].values + adf[se_col].fillna(0).values,
                        alpha=0.15,
                        color=custom_palette.get(algo, "#888"),
                        linewidth=0,
                    )

            if "model" in df.columns:
                ax.set_title(df["model"].iloc[0].split("/")[-1], fontsize=10)

            ax.set_xlabel("Epoch" if row_idx == 3 else "")
            ax.set_ylabel("Win Rate (%)" if col_idx == 0 else "", labelpad=8)
            ax.grid(True, alpha=0.15, linestyle="--")
            sns.despine(ax=ax)

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(bottom=0.09, left=0.07)

    # Section labels on the left of each row-pair
    x_label = -0.04
    for sec_idx, label in enumerate(["AlpacaEval", "MT-Bench"]):
        top = axes[sec_idx * 2, 0].get_position().y1
        bot = axes[sec_idx * 2 + 1, 0].get_position().y0
        fig.text(
            x_label,
            (top + bot) / 2,
            label,
            fontsize=13,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
            transform=fig.transFigure,
        )

    # Shared legend: all items in one row
    order = sorted(a for a in unique_algos if "PEPO $L=" in a) + [
        a for a in ["DPO", r"$\chi^2$PO", r"SFT+DPO"] if a in unique_algos
    ]
    handles = [
        plt.Line2D([0], [0], color=custom_palette[a], linewidth=2) for a in order
    ]
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
        print(f"Figure saved to {save_path}")

    plt.show()


def get_best_winrates(
    dataframes,
    y_col="winrate_initial",
    se_col="standard_error_initial",
    hue_col="algorithm",
    exclude_algos=None,
    aggregate_pepo=True,
):
    """
    Creates a summary DataFrame with the best win rate for each algorithm per model.

    Args:
        dataframes (list): List of DataFrames, one per model (from get_exp1_data).
        y_col (str): Column name for win rate.
        se_col (str, optional): Column name for standard error.
        hue_col (str): Column name for algorithm grouping.
        exclude_algos (list, optional): Algorithms to exclude.
        aggregate_pepo (bool): If True, consolidates all PEPO L variants
                               into single "PEPO" entry.

    Returns:
        pd.DataFrame: Summary table with best win rates per algorithm and model.
    """
    # Mapping for display names
    if aggregate_pepo:
        algo_map = {
            "pepo-L1": "DPO",
            "dpo": "DPO",
            "sftdpo": r"SFT+DPO",
            "chi2po": r"$\\chi^2$PO",
            "pepo-L2": "PEPO",
            "pepo-L3": "PEPO",
            "pepo-L4": "PEPO",
            r"PEPO $L=2$": "PEPO",
            r"PEPO $L=3$": "PEPO",
            r"PEPO $L=4$": "PEPO",
        }
    else:
        algo_map = {
            "pepo-L1": "DPO",
            "dpo": "DPO",
            "sftdpo": r"SFT+DPO",
            "chi2po": r"$\\chi^2$PO",
            "pepo-L2": r"PEPO $L=2$",
            "pepo-L3": r"PEPO $L=3$",
            "pepo-L4": r"PEPO $L=4$",
        }

    results = []

    for df in dataframes:
        if df.empty:
            continue

        # Get model name
        model_name = (
            df["model"].iloc[0].split("/")[-1] if "model" in df.columns else "Unknown"
        )

        # Filter and prepare data
        plot_df = df.dropna(subset=[y_col]).copy()
        if exclude_algos:
            plot_df = plot_df[~plot_df[hue_col].isin(exclude_algos)]

        plot_df["display_algo"] = plot_df[hue_col].replace(algo_map)

        # Find best (max) win rate for each algorithm
        for algo in plot_df["display_algo"].unique():
            algo_data = plot_df[plot_df["display_algo"] == algo]
            best_idx = algo_data[y_col].idxmax()
            best_row = algo_data.loc[best_idx]

            result = {
                "model": model_name,
                "algorithm": algo,
                "best_winrate": best_row[y_col],
                "best_epoch": best_row.get("epoch", None),
            }

            # Add standard error if available
            if se_col and se_col in algo_data.columns:
                result["standard_error"] = best_row.get(se_col, None)

            results.append(result)

    return pd.DataFrame(results)


def format_winrates_latex(
    summary_df,
    pivot=True,
    bold_best=True,
    include_se=True,
    font_size="small",
    short_names=True,
):
    """
    Formats the best winrates summary as a LaTeX table.

    Args:
        summary_df (pd.DataFrame): Output from get_best_winrates().
        pivot (bool): If True, creates models as rows and algorithms as columns.
        bold_best (bool): If True, bolds the best value per model.
        include_se (bool): If True, includes standard error as ±SE.
        font_size (str): Font size: 'normal', 'small', 'footnotesize', 'scriptsize'.
        short_names (bool): If True, uses abbreviated names for algorithms/models.

    Returns:
        str: LaTeX table string.
    """
    df = summary_df.copy()

    # Shorter display names for compact tables
    if short_names:
        algo_short = {
            "DPO": "DPO",
            "PEPO": "PEPO",
            r"SFT+DPO": "SFT+DPO",
            r"$\\chi^2$PO": r"$\chi^2$PO",
            r"PEPO $L=2$": r"$L{=}2$",
            r"PEPO $L=3$": r"$L{=}3$",
            r"PEPO $L=4$": r"$L{=}4$",
        }
        model_short = {
            "Llama-3.1-Tulu-3-8B-SFT": "Tulu-3-8B",
            "zephyr-7b-sft-full": "Zephyr-7B",
            "mistral-7b-sft-beta": "Mistral-7B",
            "Yi-34B-Chat": "Yi-34B",
        }
        df["algorithm"] = df["algorithm"].replace(algo_short)
        df["model"] = df["model"].replace(model_short)

    # Format values with SE
    def format_value(row):
        val = f"{row['best_winrate']:.1f}"
        if include_se and "standard_error" in row and pd.notna(row["standard_error"]):
            val += f" $\\pm$ {row['standard_error']:.1f}"
        return val

    df["formatted"] = df.apply(format_value, axis=1)

    if pivot:
        # Pivot to have algorithms as columns, models as rows
        pivot_df = df.pivot(index="model", columns="algorithm", values="best_winrate")
        formatted_pivot = df.pivot(
            index="model", columns="algorithm", values="formatted"
        )

        # Replace NaN with empty string
        formatted_pivot = formatted_pivot.fillna("")

        # Reorder columns: PEPO first, then others alphabetically
        cols = list(formatted_pivot.columns)
        pepo_cols = [c for c in cols if "PEPO" in c or c == "PEPO"]
        other_cols = sorted([c for c in cols if c not in pepo_cols])
        ordered_cols = pepo_cols + other_cols
        formatted_pivot = formatted_pivot[
            [c for c in ordered_cols if c in formatted_pivot.columns]
        ]
        pivot_df = pivot_df[[c for c in ordered_cols if c in pivot_df.columns]]

        if bold_best:
            # Bold the best value per row
            for model in pivot_df.index:
                # Skip if all values are NaN
                if pivot_df.loc[model].isna().all():
                    continue
                best_algo = pivot_df.loc[model].idxmax()
                current_val = formatted_pivot.loc[model, best_algo]
                if current_val:  # Only bold non-empty values
                    formatted_pivot.loc[model, best_algo] = f"\\textbf{{{current_val}}}"

        # Remove index/column names for cleaner output (empty corner)
        formatted_pivot.index.name = None
        formatted_pivot.columns.name = None

        # Generate LaTeX
        latex = formatted_pivot.to_latex(
            escape=False,
            column_format="l" + "c" * len(formatted_pivot.columns),
            na_rep="",
        )
    else:
        latex = df[["model", "algorithm", "formatted"]].to_latex(
            index=False,
            escape=False,
            column_format="llc",
        )

    # Wrap with font size if specified
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
    """
    Creates a LaTeX table comparing Rejection Sampling vs Token Level PEPO
    across epochs.

    Args:
        df: DataFrame from get_exp2_data().
        epochs: List of epochs to include (default: [1, 2, 3]).
        y_col: Column for win rate.
        se_col: Column for standard error.
        L: Which L value to compare (default: None, uses aggregate_best).
        aggregate_best: If True and L is None, aggregate best across
                        all L values (default: True).
        font_size: LaTeX font size.

    Returns:
        str: LaTeX table string.
    """
    if epochs is None:
        epochs = [1, 2, 3]

    df = df.copy()

    # Identify rejection sampling vs token level
    def get_method(algo):
        if "Rej." in str(algo):
            return "Rejection Sampling"
        else:
            return r"Token Level \texttt{PEPO}"

    df["method"] = df["algorithm"].apply(get_method)

    # Filter by L if specified, otherwise use all
    if L is not None:
        if "L" in df.columns:
            df = df[df["L"] == L]

    # Build table data
    rows = {}
    for method in ["Rejection Sampling", r"Token Level \texttt{PEPO}"]:
        method_df = df[df["method"] == method]
        row_data = {}
        for epoch in epochs:
            epoch_df = method_df[method_df["epoch"] == epoch]
            if len(epoch_df) > 0:
                # Take the best (max) win rate if multiple runs for same epoch
                best_idx = epoch_df[y_col].idxmax()
                winrate = epoch_df.loc[best_idx, y_col]
                se = (
                    epoch_df.loc[best_idx, se_col]
                    if se_col in epoch_df.columns
                    else None
                )
                row_data[epoch] = (winrate, se)
            else:
                row_data[epoch] = (None, None)
        rows[method] = row_data

    # Determine best per epoch
    best_per_epoch = {}
    for epoch in epochs:
        best_val = -1
        best_method = None
        for method, data in rows.items():
            winrate, _ = data.get(epoch, (None, None))
            if winrate is not None and winrate > best_val:
                best_val = winrate
                best_method = method
        best_per_epoch[epoch] = best_method

    # Format cells
    def format_cell(winrate, se, is_best):
        if winrate is None:
            return ""
        val = f"{winrate:.1f}"
        if se is not None:
            val += f" $\\pm$ {se:.2f}"
        if is_best:
            val = f"\\textbf{{{val}}}"
        return val

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(
        r"\caption{Win rates of Rejection Sampling vs. Token Level \texttt{PEPO}}"
    )
    lines.append(r"\label{tab:rej-vs-token}")
    lines.append(r"\centering")
    lines.append(f"\\{font_size}")

    col_format = "l" + "c" * len(epochs)
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\toprule")

    # Header row
    epoch_headers = " & ".join([f"epoch ${e}$" for e in epochs])
    lines.append(f" & {epoch_headers} \\\\")
    lines.append(r"\midrule")

    # Data rows
    for method in ["Rejection Sampling", r"Token Level \texttt{PEPO}"]:
        cells = []
        for epoch in epochs:
            winrate, se = rows[method].get(epoch, (None, None))
            is_best = best_per_epoch.get(epoch) == method
            cells.append(format_cell(winrate, se, is_best))
        line = f"{method} & " + " & ".join(cells) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def get_final_epoch_winrates(
    dataframes, y_col="winrate_initial", se_col=None, aggregate_pepo=True, fixed_L=None
):
    results = []

    for df in dataframes:
        if df.empty:
            continue
        model_name = df["model"].iloc[0].split("/")[-1]
        plot_df = df.dropna(subset=[y_col]).copy()

        def parse_l(algo):
            match = re.fullmatch(r"pepo-L(\d+)", str(algo))
            return int(match.group(1)) if match else None

        # Per-source algorithm final epoch
        final_by_source = {}
        for algo in plot_df["algorithm"].unique():
            adf = plot_df[plot_df["algorithm"] == algo]
            last = adf.loc[adf["epoch"].idxmax()]
            final_by_source[algo] = last

        # Display-name mapping
        if aggregate_pepo:
            algo_map = {
                "pepo-L1": "DPO",
                "dpo": "DPO",
                "sftdpo": r"SFT+DPO",
                "chi2po": r"$\chi^2$PO",
                "pepo-L2": "PEPO",
                "pepo-L3": "PEPO",
                "pepo-L4": "PEPO",
            }
        else:
            algo_map = {
                "pepo-L1": "DPO",
                "dpo": "DPO",
                "sftdpo": r"SFT+DPO",
                "chi2po": r"$\chi^2$PO",
                "pepo-L2": r"PEPO $L=2$",
                "pepo-L3": r"PEPO $L=3$",
                "pepo-L4": r"PEPO $L=4$",
            }

        # When fixed_L is set, only that PEPO L variant maps to "PEPO"
        if fixed_L is not None:
            used_L = fixed_L
            # Fallback: if fixed_L has no data for this model, pick best available
            key = f"pepo-L{fixed_L}"
            if key not in final_by_source or final_by_source[key].get("epoch", 0) == 0:
                avail = [
                    L
                    for L in [2, 3, 4]
                    if f"pepo-L{L}" in final_by_source
                    and final_by_source[f"pepo-L{L}"].get("epoch", 0) > 0
                ]
                if avail:

                    def final_val(L):
                        r = final_by_source[f"pepo-L{L}"]
                        return r[y_col] if y_col in r.index else float("-inf")

                    used_L = max(avail, key=final_val)
            for L in [2, 3, 4]:
                key = f"pepo-L{L}"
                if L == used_L:
                    algo_map[key] = "PEPO"
                elif key in algo_map:
                    del algo_map[key]

        # Group sources by display name, pick the source with highest final-epoch value
        display_groups = {}
        for source, row in final_by_source.items():
            if source not in algo_map:
                continue
            display = algo_map[source]
            display_groups.setdefault(display, []).append(row)

        for display, rows in display_groups.items():
            best = max(rows, key=lambda r: r[y_col])
            entry = {
                "model": model_name,
                "algorithm": display,
                "best_winrate": best[y_col],
                "best_epoch": best.get("epoch", None),
            }
            if se_col and se_col in best.index:
                entry["standard_error"] = best.get(se_col, None)
            results.append(entry)

    return pd.DataFrame(results)
