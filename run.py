from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.generate_id import train
from TIGER.training import train_tiger
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from utils import download_file, setup_logging
import random
import numpy as np

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:
    urls = [
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz",
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"
    ]
    print(config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    directory = "./ID_generation/preprocessing/raw_data/"
    directory_processed = "./ID_generation/preprocessing/processed/"
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory_processed, exist_ok=True)
    os.makedirs("./ID_generation/ID/", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    for url in urls:
        # Extract the filename from the URL
        filename = url.split("/")[-1]
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            print(f"{filename} not found, downloading...")
            download_file(url, filepath)
    preprocessing(config['dataset'], require_attributes=True)
    writer = setup_logging(config)
    train_config = { **config['dataset'], **{k: v for k, v in config.items() if k not in ['logging', 'dataset']} }
    features_used = "_".join(train_config['features_needed'])
    if not train_config['RQ-VAE']['original_impl']:
        save_location = train_config['saved_id_path'].replace(".pkl", f"_{features_used}_{train_config['content_model']}_{config['seed']}.pkl")
    else:
        save_location = train_config['saved_id_path'].replace(".pkl", f"_{features_used}_{train_config['content_model']}_{config['seed']}_original.pkl")
    if not os.path.exists(f'./ID_generation/ID/{save_location}'):
        print("Semantic ID file not found, Training RQ-VAE model...")
        train(train_config, device, writer)

    train_tiger(train_config, device, writer)

if __name__ == "__main__":
    main()