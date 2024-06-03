import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
import gc
from utils import WandbManager, set_weight_decay

DEFAULT_CONFIG = {
    'dataset': 'Sports_and_Outdoors',
    'RQ-VAE': {
        'batch_size': 2048,
        'epochs': 5000,
        'lr': 0.001,
        'beta': 0.25,
        'input_dim': 64,
        'hidden_dim': [2048, 1024, 512, 256],
        'latent_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'code_book_size': 256, # 'code_book_size': [4, 16, 256]
        'max_seq_len': 256,
        'val_ratio': 0.05,
        'batch_norm': True,
        'standardize': True
    }
}

def train_rqvae(model, x, device, writer, config):
    model.to(device)
    print(model)
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    beta = config["beta"]
    lr = config['lr']
    global_step = 0
    if not config['original_impl']:
        model.generate_codebook(torch.Tensor(x).to(device), device)
    if hasattr(torch.optim, config['optimizer']):
        optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), lr=lr)
        if 'weight_decay' in optimizer.param_groups[0]:
            set_weight_decay(optimizer, config['weight_decay'])
    else:
        raise NotImplementedError(f"Specified Optimizer {config['optimizer']} not implemented!!") 
    trainset, validationset = train_test_split(x, test_size=0.05, random_state=42)
    train_dataset = TensorDataset(torch.Tensor(trainset).to(device))
    val_dataset = TensorDataset(torch.Tensor(validationset).to(device))
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_quant_loss = 0.0
        total_emb_loss = 0.0
        total_commit_loss = 0.0
        total_count = 0
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            if config['original_impl']:
                count = 0
                recon_x, commitment_loss, indices = model(x_batch)
                reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
                loss = reconstruction_mse_loss + commitment_loss 
                quantization_loss = torch.Tensor([0])
                embedding_loss = torch.Tensor([0])
            else:
                recon_x, r, e, count, indices = model(x_batch)
                reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
                embedding_loss = F.mse_loss(r.detach(),e,reduction='mean')
                commitment_loss = beta*F.mse_loss(r,e.detach(),reduction='mean')
                quantization_loss = embedding_loss + commitment_loss 
                loss = reconstruction_mse_loss + quantization_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rec_loss += reconstruction_mse_loss.item()
            total_quant_loss += quantization_loss.item()
            total_emb_loss += embedding_loss.item()
            total_commit_loss += commitment_loss.item()
            total_count += count
            if isinstance(writer, WandbManager):
                codebook_indices = { f"pretrain/indices_{i}": indices[:, i] for i in range(indices.size(1)) }
                logs = {
                    "pretrain/reconstruction_loss": reconstruction_mse_loss.detach().item(),
                    "pretrain/quantization_loss": quantization_loss.detach().item(),
                    "pretrain/embedding_loss": embedding_loss.detach().item(),
                    "pretrain/commitment_loss": commitment_loss.detach().item(),
                    "pretrain/step": global_step,
                    "pretrain/epoch": epoch+1
                }
                writer.log({ **logs, **codebook_indices})
            else:
                raise NotImplementedError("Writer not implemented")

        if (epoch + 1) % 100 == 0:
            total_val_loss = 0.0
            total_val_rec_loss = 0.0
            total_val_quant_loss = 0.0
            total_val_count = 0
            total_val_emb_loss = 0.0
            total_val_comm_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    x_batch = batch[0]
                    if config['original_impl']:
                        count = 0
                        recon_x, commitment_loss, indices = model(x_batch)
                        reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
                        loss = reconstruction_mse_loss + commitment_loss 
                        quantization_loss = torch.Tensor([0])
                        embedding_loss = torch.Tensor([0])
                    else:
                        recon_x,r,e, count, indices = model(x_batch)
                        reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
                        embedding_loss = F.mse_loss(r,e,reduction='mean')
                        commitment_loss = beta*F.mse_loss(r,e,reduction='mean')
                        quantization_loss = embedding_loss + commitment_loss
                        loss = reconstruction_mse_loss + quantization_loss
                    total_val_loss += loss.item()
                    total_val_rec_loss += reconstruction_mse_loss.item()
                    total_val_quant_loss += quantization_loss.item()
                    total_val_count += count

            if isinstance(writer, WandbManager):
                codebook_indices = { f"pretrain/eval_indices_{i}": indices[:, i] for i in range(indices.size(1)) }
                logs = {
                    "pretrain/eval_reconstruction_loss": total_val_rec_loss/ len(val_dataloader),
                    "pretrain/eval_quant_loss": total_val_quant_loss/ len(val_dataloader),
                    "pretrain/eval_embedding_loss": total_val_emb_loss / len(val_dataloader),
                    "pretrain/eval_commitment_loss": total_val_comm_loss / len(val_dataloader),
                    "pretrain/epoch_unused_codebook": total_count/ len(dataloader)
                }
                writer.log({ **logs, **codebook_indices })
            else:
                raise NotImplementedError("Writer not implemented!")
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss/ len(dataloader)}, unused_codebook:{total_count/ len(dataloader)}")
            print(f"Epoch [{epoch+1}/{num_epochs}], recosntruction_loss: {total_rec_loss/ len(dataloader)}, quantization_loss: {total_quant_loss/ len(dataloader)}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {total_val_loss/ len(val_dataloader)}, unused_codebook:{total_val_count/ len(val_dataloader)}")
            print(f"Epoch [{epoch+1}/{num_epochs}], recosntruction_loss: {total_val_rec_loss/ len(val_dataloader)}, quantization_loss: {total_val_quant_loss/ len(val_dataloader)}")
    print("Training complete.")

def train(config, device, writer):
    dataset_name, save_location = config['name'], config['saved_id_path']
    content_model = config['content_model']
    model_config = config['RQ-VAE'] 
    features_used = "_".join(config["features_needed"])
    if not (os.path.exists(f'./ID_generation/preprocessing/processed/{dataset_name}_{content_model}_{features_used}_embeddings.pkl')):
        print("Embeddings not found, generating embeddings...")
        item_2_text = json.loads(open(f'./ID_generation/preprocessing/processed/{dataset_name}_{features_used}_id2meta.json').read())
        text_embedding_model = SentenceTransformer(f'sentence-transformers/{content_model}').to(device)
        item_id_2_text = {}
        for k,v in item_2_text.items():
            item_id_2_text[int(k)] = v
        sorted_text = [value for key, value in sorted(item_id_2_text.items())]
        bs = 512 if content_model == "sentence-t5-base" else 32
        with torch.no_grad():
            embeddings = text_embedding_model.encode(sorted_text, convert_to_numpy=True, batch_size=bs, show_progress_bar=True)
        #embeddings_map = {i:embeddings[i] for i in range(len(embeddings))}
        with open(f'./ID_generation/preprocessing/processed/{dataset_name}_{content_model}_{features_used}_embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
    embeddings = []
    with open(f'./ID_generation/preprocessing/processed/{dataset_name}_{content_model}_{features_used}_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    input_size = model_config['input_dim']
    hidden_sizes = model_config['hidden_dim']
    latent_size = model_config['latent_dim']
    num_levels = model_config['num_layers']
    codebook_size = model_config['code_book_size']
    dropout = model_config['dropout']
    if model_config['standardize']:
        embeddings = StandardScaler().fit_transform(embeddings)
    if model_config['pca']:
        pca = PCA(n_components=input_size, whiten=True)
        embeddings = pca.fit_transform(embeddings)
    if model_config['original_impl']:
        from .models.rqvae import RQVAE
    else:
        from .models.RQ_VAE import RQVAE
    rqvae = RQVAE(input_size, hidden_sizes, latent_size, num_levels, codebook_size, dropout)
    train_rqvae(rqvae, embeddings, device, writer, model_config)
    rqvae.to(device)
    embeddings_tensor = torch.Tensor(embeddings).to(device)
    rqvae.eval()
    if model_config['original_impl']:
        ids = rqvae.get_codes(embeddings_tensor).cpu().numpy()
        codebook_embs = torch.cat([rqvae.quantizer.codebooks[i].weight.data for i in range(len(rqvae.quantizer.codebooks))])
    else:
        ids = rqvae.encode(embeddings_tensor)
        codebook_embs = rqvae.quantization_layer.codebooks
    
    # If the ID directory does not exist, create it
    os.makedirs('./ID_generation/ID', exist_ok=True)
    
    seed = config['seed']
    save_location = f'./ID_generation/ID/{save_location.replace(".pkl", f"_{features_used}_{content_model}_{seed}")}'
    if model_config['original_impl']:
        save_location += '_original'
    if model_config['pca']:
        save_location += '_pca'
    save_location += f"_{model_config['optimizer']}"                    
    if not model_config['original_impl']:
        torch.save(codebook_embs, f'{save_location}.pth')
        with open(f'{save_location}.pkl', 'wb') as f:
            pickle.dump(ids, f)
    else:
        torch.save(codebook_embs, f'{save_location}.pth')
        with open(f'{save_location}.pkl', 'wb') as f:
            pickle.dump(ids, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RQ-VAE Model")

    # General settings
    parser.add_argument("--dataset", type=str, default=DEFAULT_CONFIG['dataset'], help="Dataset name")
    parser.add_argument("--save_location", type=str, default="Sports_and_Outdoors_semantic_id0.pkl", help="Saved File Name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Saved File Name")
    # RQ-VAE settings
    rq_vae_config = DEFAULT_CONFIG['RQ-VAE']
    parser.add_argument("--batch_size", type=int, default=rq_vae_config['batch_size'], help="Batch size")
    parser.add_argument("--epochs", type=int, default=rq_vae_config['epochs'], help="Number of epochs")
    parser.add_argument("--lr", type=float, default=rq_vae_config['lr'], help="Learning rate")
    parser.add_argument("--beta", type=float, default=rq_vae_config['beta'], help="Beta")
    parser.add_argument("--input_dim", type=int, default=rq_vae_config['input_dim'], help="Input dimension")
    parser.add_argument("--hidden_dim", type=json.loads, default=rq_vae_config['hidden_dim'], help="Hidden dimensions")
    parser.add_argument("--latent_dim", type=int, default=rq_vae_config['latent_dim'], help="Latent dimension")
    parser.add_argument("--num_layers", type=int, default=rq_vae_config['num_layers'], help="Number of layers")
    parser.add_argument("--dropout", type=float, default=rq_vae_config['dropout'], help="Dropout rate")
    parser.add_argument("--code_book_size", type=json.loads, default=rq_vae_config['code_book_size'], help='Code book size')
    parser.add_argument("--max_seq_len", type=int, default=rq_vae_config['max_seq_len'], help="Maximum sequence length")
    parser.add_argument("--val_ratio", type=float, default=rq_vae_config['val_ratio'], help="Validation ratio")
    parser.add_argument("--standardize", action="store_false", default = True, help="Standardize the embeddings before PCA")
    parser.add_argument("--no_standardize", action="store_false", help="Do not standardize the embeddings before PCA")

    args = parser.parse_args()

    # The train function call will now include args.dataset and a dictionary of RQ-VAE settings derived from args
    rq_vae_settings = {k: v for k, v in vars(args).items() if k in rq_vae_config}
    train(args.dataset, args.save_location, rq_vae_settings, args.device)
    torch.cuda.empty_cache()
    gc.collect()