import os
from tqdm import tqdm
import numpy as np
from transformers import T5Config,T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch
from transformers.optimization import get_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from .evaluation import evaluate, evaluate_cold_start
from .load_data import load_data, load_data_cold_start
from utils import WandbManager, get_lr, CustomDataset
import uuid
    
def initialize_embeddings(model, save_location):
    # initialize embeddings of the transformer with embeddings coming from RQ-VAE
    # start-index = 1
    # codebook1: 1-256, codebook 2: 257-512 codebook 3: 513-768
    model_embs = model.shared.weight.data
    codebook_embs = torch.load(f'./ID_generation/ID/{save_location.replace(".pkl", ".pth")}')
    codebook_embs = torch.cat([p for p in codebook_embs])
    assert model_embs.shape[-1] == codebook_embs.shape[-1], "Embeddings of RQ-VAE must be of equal dimension to transformer embeddings"
    model_embs[1:769] = codebook_embs
    model.shared.weight.data = model_embs

def train_tiger(config, device, writer):
    dataset, save_location, cold_start, seed = config['name'], config['saved_id_path'], config['cold_start'], config['seed'] 
    max_items_per_seq, content_model, features_used = config['max_items_per_seq'], config['content_model'], config["features_needed"]
    features_used = "_".join(features_used)

    save_location = save_location.replace(".pkl", f"_{features_used}_{content_model}_{seed}")
    if config['RQ-VAE']['original_impl']:
        save_location += '_original'
    if config['RQ-VAE']['pca']:
        save_location += '_pca'
    save_location += f"_{config['RQ-VAE']['optimizer']}"
    save_location = f'{save_location}.pkl'
    config = config['TIGER']
    if cold_start:
        training_data, val_data, test_data, val_unseen_semantic_ids, test_unseen_semantic_ids = load_data_cold_start(dataset, save_location, 
                                                                                                                     max_items_per_seq=max_items_per_seq)
    else:
        training_data, val_data, test_data = load_data(dataset, save_location, max_items_per_seq=max_items_per_seq)

    train_dataset = CustomDataset(training_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)
    t5_config = config['T5']
    trainer_config = config['trainer']
    model_config = T5Config(
        num_layers=t5_config['encoder_layers'], 
        num_decoder_layers=t5_config['decoder_layers'],
        d_model=t5_config['d_model'],
        d_ff=t5_config['d_ff'],
        num_heads=t5_config['num_heads'],
        d_kv=t5_config['d_kv'],  # Size of the key, query, value projections per attention head.
        dropout_rate=t5_config['dropout_rate'],
        activation_function=t5_config['activation_function'],
        vocab_size=3026,  # liu: 1024 + 2000 + 1
        pad_token_id=0,
        eos_token_id=3025,
        decoder_start_token_id=0,
        feed_forward_proj=t5_config['feed_forward_proj'],
        n_positions=config['n_positions'],
    )
    
    output_path = str(uuid.uuid4()).split('-')[0]
    os.makedirs(f"./outputs/{output_path}/logs", exist_ok=True)
    os.makedirs(f"./outputs/{output_path}/results", exist_ok=True)
    
    # Initialize the model with the custom configuration
    model = T5ForConditionalGeneration(config=model_config).to(device)
    if t5_config["initialize_pretrained"]:
        initialize_embeddings(model, save_location)
    
    total_steps = trainer_config['steps']
    batch_size = trainer_config['batch_size']
    learning_rate = trainer_config['lr']
    eval_batch_size = trainer_config['eval_batch_size']
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=trainer_config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler()
    
    scheduler = get_scheduler(
        name=trainer_config["scheduler"],
        optimizer=optimizer,
        num_warmup_steps=trainer_config['warmup_steps'],
        num_training_steps=total_steps,
    )

    model.train()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print(f"Total number of epochs{int(np.ceil(total_steps / len(train_dataloader)))}")
    best_ndcg_10 = 0.0
    global_step = 0
    best_epoch = 0
    for epoch in range(int(np.ceil(total_steps / len(train_dataloader)))):
        progress_bar = tqdm(range(len(train_dataloader)))
        total_loss = 0.0
        batch_num = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            batch_num += 1
            global_step += 1
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {(total_loss/batch_num):.4f}")
            progress_bar.update(1)
            if isinstance(writer, WandbManager):
                logs = {
                    'train/loss': loss.item(),
                    'train/step': global_step,
                    'train/epoch': epoch+1,
                    'train/lr': get_lr(optimizer)
                }
                writer.log(logs)
            else:
                raise NotImplementedError("Writer not implemented!!")

        progress_bar.close()
        if (epoch + 1) % 2 == 0:
            if cold_start:
                recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate_cold_start(model, val_dataloader, val_unseen_semantic_ids, device)
            else:
                recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate(model, val_dataloader, device)
            if isinstance(writer, WandbManager):
                logs = {
                    'eval/validation_loss': eval_loss,
                    'eval/Recall@5': recall_5,
                    'eval/Recall@10': recall_10,
                    'eval/NDCG@5': ndcg_5,
                    'eval/NDCG@10': ndcg_10,
                    'train/step': global_step
                }
                writer.log(logs)
            else:
                raise NotImplementedError("Writer not implemented!!")
            if ndcg_10 > best_ndcg_10:
                best_ndcg_10 = ndcg_10
                best_epoch = epoch
                model.to('cpu')
                torch.save(model.state_dict(), f"./outputs/{output_path}/results/tiger_best_exp_{dataset}_{seed}.pt")
                model.to(device)
            # Testing as well, to see if it is overfitting issues or test performance never goes up
            if cold_start:
                recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate_cold_start(model, test_dataloader, test_unseen_semantic_ids, device, num_beams=30)
            else:
                recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate(model, test_dataloader, device, num_beams=30)
            if isinstance(writer, WandbManager):
                logs = {
                    'test/loss': eval_loss,
                    'test/Recall@5': recall_5,
                    'test/Recall@10': recall_10,
                    'test/NDCG@5': ndcg_5,
                    'test/NDCG@10': ndcg_10
                }
                writer.log(logs)

        #if (epoch + 1) % 20 == 0:
        #    model.to('cpu')
        #    torch.save(model.state_dict(), f"./results/tiger_epoch_{epoch+1}_exp{config['exp_id']}.pt")
        #    model.to(device)
        if (best_epoch + trainer_config["patience"] < epoch) and global_step > trainer_config["warmup_steps"]:
            break
    print("Testing...")
    model.load_state_dict(torch.load(f"./outputs/{output_path}/results/tiger_best_exp_{dataset}_{seed}.pt"))
    if cold_start:
        recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate_cold_start(model, test_dataloader, test_unseen_semantic_ids, device, num_beams=30)
    else:
        recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate(model, test_dataloader, device, num_beams=30)
    if isinstance(writer, WandbManager):
        logs = {
            'test/loss': eval_loss,
            'test/Recall@5': recall_5,
            'test/Recall@10': recall_10,
            'test/NDCG@5': ndcg_5,
            'test/NDCG@10': ndcg_10
        }
        writer.log(logs)
    else:
        raise NotImplementedError("Writer not implemented!!")
    writer.close()
