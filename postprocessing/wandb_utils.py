import wandb
import pandas as pd
import collections
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta

def collect_run_summaries(runs):
    """
    Collects summary metrics from all the runs and returns a dataframe with one row per run.

    Args:
        runs: Wandb Runs object.

    Returns: pd.DataFrame

    """
    all_dicts = []
    for run in tqdm(runs):
        summary = run.summary._json_dict
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        all_items = {"run_name": run.name, **summary, **config}

        run_dict = {}
        # add k-v pairs for current run
        for k, v in all_items.items():
            if isinstance(v, dict):
                d = pd.json_normalize(v, sep='.').to_dict(orient="records")[0]
                for k1, v1 in d.items():
                    run_dict[k1] = v1
            else:
                run_dict[k] = v
        all_dicts.append(run_dict)
    df = pd.DataFrame(all_dicts)
    return df


def write_run_summaries(runs, save_dir="./results"):
    df = collect_run_summaries(runs)
    save_path = Path(save_dir) / datetime.now().strftime("%d-%m-%Y_%Hh%Mm")
    save_path.mkdir(exist_ok=True, parents=True)
    df.to_csv(save_path / "summaries.csv")


def collect_run_histories(runs):
    """
    Collects all the metrics from all the runs and returns a dataframe with one step per row.

    Args:
        runs: Wandb Runs object.

    Returns: pd.DataFrame

    """
    dicts_per_experiment = collections.defaultdict(list)
    for run in tqdm(runs):
        if run.state == "running":
            continue
        history = run.scan_history()
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        exp_name = config["experiment_id"]
        run_dict, all_items = {}, {"run_name": run.name, **config}
        for k, v in all_items.items():
            try:
                v = eval(v)
            except:
                pass
            if isinstance(v, dict):
                if v == {}:
                    continue
                d = pd.json_normalize(v, sep='.').to_dict(orient="records")[0]
                for k1, v1 in d.items():
                    run_dict[k1] = v1
            else:
                run_dict[k] = v
        for row in history:
            row = {k: v for k, v in row.items() if not k.startswith("parameters/") and not k.startswith("gradients/")}
            dicts_per_experiment[exp_name].append({**run_dict, **row})
    df_per_experiment = {name: pd.DataFrame(d) for name, d in dicts_per_experiment.items()}
    return df_per_experiment


def write_run_histories(runs, save_dir="./results"):
    # Collecting histories for all runs results in huge DataFrame
    # better to collect them separately and write files per experiment_name
    print("Writing run histories to: ", save_dir)
    df_per_experiment = collect_run_histories(runs)
    timestamp = datetime.now().strftime("%d-%m-%Y_%Hh%Mm")
    for exp_name, df in df_per_experiment.items():
        save_path = Path(save_dir) / timestamp / exp_name
        save_path.mkdir(exist_ok=True, parents=True)
        df.to_csv(save_path / "histories.csv")


def load_run_histories(exp_names, dir_path="./results/histories", api=False, force_reload=False,
                       runs_dir="ml_eva/EVA", cache_dir="./cache_dir", exp_name=None,
                       task_name=None):
    """
    Loads histories for a given experiment name either from local directory or directly from wandb API.

    Args:
        exp_names: Str or List.
        dir_path: Str. Path to the local directory with histories.
        api: Bool. If True, histories will be loaded from wandb API directly.

    Returns: pd.DataFrame

    """
    if not isinstance(exp_names, list):
        exp_names = [exp_names]
    if api:
        # automatically caches histories to avoid repeated API calls
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        exp_name = exp_name.replace("/", "_") if exp_name is not None else None
        cache_file = cache_dir / f"{exp_name}.csv"
        if not force_reload and exp_name is not None: 
            if cache_file.exists():
                print(f"Loading histories from cache: {cache_file}")
                return pd.read_csv(str(cache_file), index_col=0)
                
        filters = {"$or": [{"config.experiment_id": name} for name in exp_names]}
        runs = wandb.Api(timeout=50).runs(runs_dir, filters=filters)
        # filter by experiment_name
        df = pd.concat([d for d in collect_run_histories(runs).values()])
        df.to_csv(str(cache_file))
    else:
        exp_paths = Path(dir_path).glob('**/*.csv')
        df = pd.concat([pd.read_csv(p, index_col=0) for p in exp_paths if p.parent.name in exp_names])
    return df
        
