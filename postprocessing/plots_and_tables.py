import argparse
import os
import warnings
import numpy as np
from pathlib import Path
from scipy.stats import sem

import pandas as pd

from wandb_utils import load_run_histories
from exps import load_exp_config


def filter_df(df, filter):
    df_temp = df
    for key, vals in filter.items():
        vals = [vals] if not isinstance(vals, list) else vals
        df_temp = df_temp[df_temp[key].isin(vals)]
    return df_temp


def aggregate_results(df, step, round=2, metric_col=["eval/accuracy"], step_col="train/epoch", seed_col="seed",
                      groupby=("env", "agent"), agg_cols=("mean", "std"), drop_first_step=False,
                      agg_over_tasks=False, sortby_col=None, save_dir=None, other_metrics=None, 
                      agg_over_cols=None, param_counts=None, proxy_steps_of_interest=None, dump_best=False):
    if not isinstance(groupby, list):
        groupby = list(groupby)
    if not isinstance(agg_cols, list):
        agg_cols = list(agg_cols)
    if agg_cols == ["mean", "ci95"]: 
        agg_cols = ["mean", lambda x: 1.96 * np.std(x) / np.sqrt(len(x))]
    if agg_cols == ['mean', 'sem']:
        agg_cols = ['mean', sem]

    if proxy_steps_of_interest is not None: 
        # for all experiments in proxy_steps_drop_exps,
        # shift all steps to left by proxy_step_sub
        exps, sub = proxy_steps_of_interest["exps"], proxy_steps_of_interest["sub"]
        df = df.copy()
        df.loc[df["experiment_id"].isin(exps), step_col] -= sub

    # sanity check
    if step == "max":
        df_temp = df.groupby(groupby + [seed_col], dropna=False)[metric_col].max().reset_index()
        df_temp = df[groupby + metric_col + [step_col, seed_col]]
    else:
        raise NotImplementedError()

    if other_metrics is not None:
        if not isinstance(other_metrics, list):
            other_metrics = [other_metrics]
        metric_col = [metric_col] + other_metrics

    if drop_first_step:
        df_temp = df_temp[~df_temp.step.isin([0, 1])]

    for m_col in metric_col:
        df_temp[m_col] = pd.to_numeric(df_temp[m_col])
    # take the max for each seed, i.e., filter out failed runs for a particular seed
    df_temp_max = df_temp.groupby(groupby + [seed_col], dropna=False)[metric_col].max().reset_index()

    if agg_over_tasks: 
        # aggregate
        groupby_cols = [c for c in groupby if c != "task_name"] + [seed_col]
        df_temp_max = df_temp_max.groupby(groupby_cols, dropna=False).agg(["mean"]).reset_index().droplevel(1, axis=1)
    
    df_temp_max = df_temp_max.drop(seed_col, axis=1)
    
    if agg_over_cols is not None:
        groupby = [c for c in groupby if c not in agg_over_cols]

    # do aggregation over all seeds
    df_temp_agg = df_temp_max.groupby(groupby, dropna=False).agg(agg_cols).reset_index()
    round_cols = [c for c in df_temp_agg.columns if c[0] not in groupby]
    df_temp_agg[round_cols] = df_temp_agg[round_cols].round(round)

    if sortby_col:
        df_temp_agg = df_temp_agg.sort_values(sortby_col)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        df_temp_agg.to_excel(save_dir / "aggregate.xlsx")

    if param_counts is not None:
        df_temp_agg["Param Counts"] = param_counts

    return df_temp_agg


def drop_col_level(df_agg):
    df_agg_readable = df_agg.droplevel(1, axis=1)
    new_cols, agg_cols = [], []
    for multi in df_agg.columns:
        if all(m != "" for m in multi):
            new_col = "/".join(multi).rstrip("/")
            agg_cols.append(new_col)
        else:
            new_col = multi[0]
        new_cols.append(new_col)
    df_agg_readable.columns = new_cols
    return df_agg_readable, agg_cols


def convert_column_names(df):
    col_to_name = {
        "trainer.lr": "TIGER.lr",
        "T5.initialize_pretrained": "initialize_embs"
    }
    df.columns = [col_to_name.get(c, c) for c in df.columns]
    return df


def convert_agg_cols_to_readable(df, drop_std=False, add_avg=False, avg_over_metrics=None):
    df_temp, agg_cols = drop_col_level(df)

    if add_avg:
        # compute mean+-std for mean cols
        mean_and_stds = [f"{df_temp[c].mean().round(2)} \u00B1 {df_temp[c].std().round(2)}"
                         for c in df_temp.columns if "/mean" in c]

    if drop_std:
        # only preserve mean columns, assumes that there are (mean, std)-pairs
        df_temp = df_temp.drop([c for c in agg_cols if "/std" in c], axis=1)
        # set new col names
        df_temp.columns = [c.rsplit("/", 1)[0] for c in df_temp.columns]
    else:
        # combine mean and std columns to one column "mean +- std"
        df_temp[agg_cols] = df_temp[agg_cols].astype(str)
        for c1, c2 in zip(agg_cols[::2], agg_cols[1::2]):
            c = c1.rsplit("/", 1)[0]
            df_temp[c] = df_temp[[c1, c2]].agg(" \u00B1 ".join, axis=1)
            df_temp.drop([c1, c2], axis=1, inplace=True)
    
    if avg_over_metrics is not None: 
        for metric in avg_over_metrics: 
            means = df_temp[[c for c in df_temp.columns if metric in c]].mean(axis=1).round(2)
            stds = df_temp[[c for c in df_temp.columns if metric in c]].std(axis=1).round(2)
            df_temp[f"averaged{metric}"] = [f"{m} \u00B1 {s}" for m, s in zip(means, stds)]
            df_temp.drop([c for c in df_temp.columns if metric in c and c != f"averaged{metric}"], axis=1, inplace=True)
                        
    if add_avg:
        # needs to be done extra as otherwise would consider wrong std
        df_temp.loc[len(df_temp)] = ["Average"] + mean_and_stds

    df_temp = convert_column_names(df_temp)

    return df_temp


def make_pivot_table(df, index, columns):
    index, cols = list(index), list(columns)
    df_temp = df
    df_temp = df_temp.pivot(index=index, columns=cols)
    df_temp.columns = df_temp.columns.droplevel()
    df_temp.columns.name, df_temp.index.name = None, None
    return df_temp


def write_table(df, save_dir, exp_name=None, transpose=True, drop_idx=False, escape=False, postfix=""):
    save_dir = Path(save_dir)
    if exp_name is not None:
        save_dir = save_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    df_temp = df
    if transpose:
        df_temp = df_temp.T
    df_temp.to_latex(save_dir / f"scores{postfix}.tex", escape=escape, index=not drop_idx)
    df_temp.to_csv(save_dir / f"scores{postfix}.csv", index=not drop_idx)


def generate_table(df, escape=False, save_dir=None, table_kwargs=None, exp_name=None, postfix="", dump_best=False):
    if table_kwargs is None:
        table_kwargs = {}
    # additional kwargs
    transpose = table_kwargs.pop("transpose", True)
    drop_std = table_kwargs.pop("drop_std", False)
    add_avg = table_kwargs.pop("add_avg", False)
    drop_idx = table_kwargs.pop("drop_idx", False)
    pivot_kwargs = table_kwargs.pop("pivot_kwargs", {})
    avg_over_metrics = table_kwargs.pop("avg_over_metrics", None)

    # aggregate results
    df_agg = aggregate_results(df, **table_kwargs, dump_best=dump_best)
    df_agg = convert_agg_cols_to_readable(df_agg, drop_std=drop_std, add_avg=add_avg, avg_over_metrics=avg_over_metrics)
    if pivot_kwargs:
        df_agg = make_pivot_table(df_agg, **pivot_kwargs)
    if save_dir is not None:
        write_table(df_agg, save_dir, exp_name=exp_name, transpose=transpose,
                    drop_idx=drop_idx, escape=escape, postfix=postfix)
        if dump_best:
            np.save(Path(save_dir) / f"{exp_name}_score_dict.npy", score_dict)
    return df_agg


def generate_tables(df, escape=False, save_dir=None, table_kwargs=None, exp_name=None, table_filters=None,
                    dump_best=False):
    if table_filters is not None:
        tables = []
        for i, filter in enumerate(table_filters):
            df_temp = filter_df(df, filter)
            df_temp = generate_table(df_temp, escape=escape, save_dir=save_dir,
                                     table_kwargs=table_kwargs.copy(), exp_name=exp_name, postfix=f"_{i}",
                                     dump_best=dump_best)
            tables.append(df_temp)
        return tables
    return generate_table(df, escape=escape, save_dir=save_dir, table_kwargs=table_kwargs, exp_name=exp_name,
                          dump_best=dump_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--force_reload", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--runs_dir", type=str, default="instruct_genret/genrec")
    parser.add_argument("--show_test", action="store_true")
    parser.add_argument("--dir_path", type=str, default="./wandb/test")
    parser.add_argument("--tab_dir", type=str, default="./tables")
    parser.add_argument("--dump_best", action="store_true")
    args = parser.parse_args()

    table_kwargs, table_filters = {}, {}
    if args.exp_name is not None:
        exp_names, table_kwargs, table_filters = load_exp_config(args.exp_name)
    else:
        exp_names = args.exp_names.split(",")

    if args.show_test:
        table_kwargs["metric_col"] = [col.replace('eval', 'test') for col in table_kwargs['metric_col']]
    # collect histories
    df = load_run_histories(exp_names, dir_path=args.dir_path, api=args.api, runs_dir=args.runs_dir,
                            force_reload=args.force_reload, exp_name=args.exp_name)
    if args.table:
        # Regular performance tables
        df_tab = generate_tables(df, save_dir=args.tab_dir, table_kwargs=table_kwargs, exp_name=args.exp_name,
                                 table_filters=table_filters, dump_best=args.dump_best)
        print(df_tab)
