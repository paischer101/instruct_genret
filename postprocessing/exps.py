# measured in train/epochs - unfortunately this is different across tasks

TIGER = {
    "reproducing_beauty": {
        "experiment_name": ["reproducing_beauty"],
        "table_kwargs": {
            "step": "max",
            "metric_col": ["eval/Recall@5", "eval/Recall@10", "eval/NDCG@5", "eval/NDCG@10"],
            "groupby": ["name", "trainer.lr", "T5.initialize_pretrained", "T5.dropout_rate", "trainer.patience", "max_items_per_seq"],
            "sortby_col": ["name", "trainer.lr", "T5.initialize_pretrained", "T5.dropout_rate", "trainer.patience", "max_items_per_seq"],
            "add_avg": False,
            "transpose": False,
            "drop_idx": True,
            "round": 5
        },
    },
    "reproducing_sports": {
        "experiment_name": ["reproducing_sports"],
        "table_kwargs": {
            "metric_col": ["eval/Recall@5", "eval/Recall@10", "eval/NDCG@5", "eval/NDCG@10"],
            "step": "max",
            "groupby": ["name", "trainer.lr", "T5.initialize_pretrained", "T5.dropout_rate", "trainer.patience", "max_items_per_seq"],
            "sortby_col": ["name", "trainer.lr", "T5.initialize_pretrained", "T5.dropout_rate", "trainer.patience", "max_items_per_seq"],
            "add_avg": False,
            "transpose": False,
            "drop_idx": True,
            "round": 5
        },
    },
    "reproducing_toys": {
        "experiment_name": ["reproducing_toys"],
        "table_kwargs": {
            "step": "max",
            "metric_col": ["eval/Recall@5", "eval/Recall@10", "eval/NDCG@5", "eval/NDCG@10"],
            "agg_cols": ['mean', 'sem'],
            "groupby": ["name", "trainer.lr", "T5.initialize_pretrained", "trainer.scheduler", "trainer.warmup_steps", "trainer.patience", "T5.dropout_rate"],
            "sortby_col": ["name", "trainer.lr", "T5.initialize_pretrained", "trainer.scheduler", "trainer.warmup_steps", "trainer.patience", "T5.dropout_rate"],
            "add_avg": False,
            "transpose": False,
            "drop_idx": True,
            "round": 5
        },
    },
    "better_rqvae": {
        "experiment_name": ["better_rqvae"],
        "table_kwargs": {
            "step": "max",
            "metric_col": ["eval/Recall@5", "eval/Recall@10", "eval/NDCG@5", "eval/NDCG@10"],
            "agg_cols": ['mean', 'sem'],
            "groupby": ["name", "trainer.lr", "content_model", "trainer.patience", "features_needed", "T5.d_model", "T5.encoder_layers"],
            "sortby_col": ["name", "trainer.lr", "content_model", "trainer.patience", "features_needed", "T5.d_model", "T5.encoder_layers"],
            "add_avg": False,
            "transpose": False,
            "drop_idx": True,
            "round": 5
        },
    },
    "residual_rqvae": {
        "experiment_name": ["residual_rqvae"],
        "table_kwargs": {
            "step": "max",
            "metric_col": ["eval/Recall@5", "eval/Recall@10", "eval/NDCG@5", "eval/NDCG@10"],
            "agg_cols": ['mean', 'sem'],
            "groupby": ["name", "trainer.lr", "content_model", "trainer.patience", "features_needed", "max_items_per_seq", "T5.encoder_layers"],
            "sortby_col": ["name", "trainer.lr", "content_model", "trainer.patience", "features_needed", "max_items_per_seq", "T5.encoder_layers"],
            "add_avg": False,
            "transpose": False,
            "drop_idx": True,
            "round": 5
        },
    }
}

# add additional experiments here for extracting validation metrics!

ALL = {**TIGER}


def load_exp_config(exp_name):
    assert exp_name in ALL, "Unknown experiment configuration"
    exp_names = ALL[exp_name].get("experiment_name", {})
    table_kwargs = ALL[exp_name].get("table_kwargs", {})
    table_filters = ALL[exp_name].get("table_filters", None)
    return exp_names, table_kwargs, table_filters
