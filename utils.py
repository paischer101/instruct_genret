import requests
import os
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.data.items()}
        return item

class WandbManager:
    def __init__(self) -> None:
        assert self._wandb_available, "wandb is not installe, please install via pip install wandb"
        import wandb
        self._wandb = wandb
        self._initialized = False

    def _wandb_available():
        # any value of WANDB_DISABLED disables wandb
        if os.getenv("WANDB_DISABLED", "").upper():
            print(
                "Not using wandb for logging, if this is not intended, unset WANDB_DISABLED env var"
            )
            return False
        return importlib.util.find_spec("wandb") is not None
    
    def setup(self, args, **kwargs):
        if not isinstance(args, dict):
            args = args.__dict__
        project_name = args.get("project", "debug")

        combined_dict = { **args, **kwargs}
        self._wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            entity=args.get("entity", None),
            # track hyperparameters and run metadata
            config=combined_dict
        )
        self._initialized = True

    def log(self, logs):
        self._wandb.log(logs)

    def close(self):
        pass
        
    def summarize(self, outputs):
        # add values to the wandb summary => only works for scalars
        for k, v in outputs.items():
            self._wandb.run.summary[k] = v.item()
            
def download_file(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {os.path.basename(path)}")
    else:
        print(f"Failed to download {os.path.basename(path)}")

def setup_logging(config):
    logger_conf = config['logging']
    model_config = config['dataset']
    if logger_conf["writer"] == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"./logs/tiger_exp{config['exp_id']}")
    elif logger_conf["writer"] == "wandb":
        if logger_conf['mode'] == "offline":
            os.environ['WANDB_MODE'] = "offline"
        from utils import WandbManager
        writer = WandbManager()
        writer.setup({ **logger_conf, **model_config, 'experiment_id': config['experiment_id'], 'seed': config['seed'] })
    else:
        raise NotImplementedError("Specified writer not recognized!")
    return writer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']