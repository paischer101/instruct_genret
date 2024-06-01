import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import os
parent_dir = os.path.dirname('/'.join(os.path.realpath(__file__).split("/")[:-1]))
sys.path.append(parent_dir)
from utils import CustomDataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from transformers import T5Config, T5ForConditionalGeneration

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--semantic_id_path', type=str, required=True)
    parser.add_argument('--cold_start', action='store_true')
    parser.add_argument('--dataset', choices=['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games'])
    parser.add_argument('--batch_size', type=int, default=256)
    return parser.parse_args()

def dcg(scores):
    """Compute the Discounted Cumulative Gain."""
    scores = np.asfarray(scores)  # Ensure scores is an array of floats
    return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))

def ndcg_at_k(r, k):
    """Compute NDCG at rank k."""
    r = np.asfarray(r)[:k]  # Ensure r is an array of floats and take top k scores
    dcg_max = dcg(sorted(r, reverse=True))
    if not dcg_max:
        return 0.
    return dcg(r) / dcg_max

def calculate_metrics(outputs, labels):
    batch_size, k, _ = outputs.shape  # Assuming outputs is [batch_size, 10, seq_len]
    recall_at_5, recall_at_10 = [], []
    ndcg_at_5, ndcg_at_10 = [], []

    for i in range(batch_size):
        label = labels[i].unsqueeze(0)  # [1, seq_len]
        out = outputs[i]  # [10, seq_len]

        matches = torch.all(torch.eq(out.unsqueeze(1), label.unsqueeze(0)), dim=2)  # [10, 1, seq_len] -> [10, 1]
        matches = matches.any(dim=1).cpu().numpy()  # [10]

        # Recall
        recall_at_5.append(matches[:5].sum() / 1.0)  # Assuming each label has only 1 correct match.
        recall_at_10.append(matches.sum() / 1.0)

        # NDCG (binary relevance)
        ndcg_at_5.append(ndcg_at_k(matches, 5))
        ndcg_at_10.append(ndcg_at_k(matches, 10))

    # Calculate mean metrics
    metrics = (
        np.mean(recall_at_5),
        np.mean(recall_at_10),
        np.mean(ndcg_at_5),
        np.mean(ndcg_at_10),
    )

    return metrics

def evaluate(model, dataloader, device, num_beams=10):
    model.eval()
    recall_at_10 = []
    recall_at_5s = []
    recall_at_10s = []
    ndcg_at_5s = []
    ndcg_at_10s = []
    losses = []
    progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        batch_size = batch['input_ids'].shape[0]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(outputs.loss.item())
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256, num_beams=num_beams, num_return_sequences=10)
        outputs = outputs[:, 1:5].reshape(batch_size, 10, -1)
        labels = labels[:,:-1]
        recall_at_5, recall_at_10, ndcg_at_5, ndcg_at_10= calculate_metrics(outputs, labels)
        recall_at_5s.append(recall_at_5)
        recall_at_10s.append(recall_at_10)
        ndcg_at_5s.append(ndcg_at_5)
        ndcg_at_10s.append(ndcg_at_10)
        progress_bar.set_description(f"recall@10: {(sum(recall_at_10s) / len(recall_at_10s)):.4f}, NDCG@10: {(sum(ndcg_at_10s) / len(ndcg_at_10s)):.4f}")
        progress_bar.update(1)
    progress_bar.close()
    print(f"Validation Loss: {sum(losses) / len(losses)}")
    print(f"recall@5: {sum(recall_at_5s) / len(recall_at_5s)}")
    print(f"recall@10: {sum(recall_at_10s) / len(recall_at_10s)}")
    print(f"NDCG@5: {sum(ndcg_at_5s) / len(ndcg_at_5s)}")
    print(f"NDCG@10: {sum(ndcg_at_10s) / len(ndcg_at_10s)}")
    model.train()
    return sum(recall_at_5s) / len(recall_at_5s), sum(recall_at_10s) / len(recall_at_10s), sum(ndcg_at_5s) / len(ndcg_at_5s), sum(ndcg_at_10s) / len(ndcg_at_10s), sum(losses) / len(losses)


def evaluate_cold_start(model, dataloader, unseen_semantic_ids, device, num_beams=10, top_k=5, epsilon=0.1):
    model.eval()
    recall_at_10 = []
    recall_at_5s = []
    recall_at_10s = []
    ndcg_at_5s = []
    ndcg_at_10s = []
    losses = []
    progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        batch_size = batch['input_ids'].shape[0]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(outputs.loss.item())
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256, num_beams=num_beams, num_return_sequences=top_k)
        outputs = outputs[:, 1:5].reshape(batch_size, top_k, -1)
        unique_first_ids = torch.unique(outputs[:, :-1], dim=1)
        outputs = outputs.numpy().tolist()
        import pdb
        pdb.set_trace()
        for sample in range(batch_size):
            for id in unique_first_ids[batch_size]:
                matches = np.where(id == unseen_semantic_ids[:, :-1]).all(axis=-1)
                # select only the epsilon fraction of unseen ids that match
                matches = np.random.choice(matches, size=int(epsilon*len(matches)), replace=False)
                outputs[sample].extend(unseen_semantic_ids[matches].tolist())
        labels = labels[:,:-1]
        recall_at_5, recall_at_10, ndcg_at_5, ndcg_at_10= calculate_metrics(outputs, labels)
        recall_at_5s.append(recall_at_5)
        recall_at_10s.append(recall_at_10)
        ndcg_at_5s.append(ndcg_at_5)
        ndcg_at_10s.append(ndcg_at_10)
        progress_bar.set_description(f"recall@10: {(sum(recall_at_10s) / len(recall_at_10s)):.4f}, NDCG@10: {(sum(ndcg_at_10s) / len(ndcg_at_10s)):.4f}")
        progress_bar.update(1)
    progress_bar.close()
    print(f"Validation Loss: {sum(losses) / len(losses)}")
    print(f"recall@5: {sum(recall_at_5s) / len(recall_at_5s)}")
    print(f"recall@10: {sum(recall_at_10s) / len(recall_at_10s)}")
    print(f"NDCG@5: {sum(ndcg_at_5s) / len(ndcg_at_5s)}")
    print(f"NDCG@10: {sum(ndcg_at_10s) / len(ndcg_at_10s)}")
    model.train()
    return sum(recall_at_5s) / len(recall_at_5s), sum(recall_at_10s) / len(recall_at_10s), sum(ndcg_at_5s) / len(ndcg_at_5s), sum(ndcg_at_10s) / len(ndcg_at_10s), sum(losses) / len(losses)

def main(args):
    model_config = dict(OmegaConf.load(f"./configs/dataset/{args.dataset}.yaml"))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    from load_data import load_data, load_data_cold_start
    t5_config = model_config['TIGER']['T5']
    if args.cold_start:
        _, _, test_data, _, test_unseen_semantic_ids = load_data_cold_start(args.dataset, args.semantic_id_path)
    else:
        _, _, test_data = load_data(args.dataset, args.semantic_id_path)

    test_dataset = CustomDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model_config = T5Config(
        num_layers=t5_config['encoder_layers'], 
        num_decoder_layers=t5_config['decoder_layers'],
        d_model=t5_config['d_model'],
        d_ff=t5_config['d_ff'],
        num_heads=t5_config['num_heads'],
        d_kv=t5_config['d_kv'],
        dropout_rate=t5_config['dropout_rate'],
        activation_function=t5_config['activation_function'],
        vocab_size=3026,
        pad_token_id=0,
        eos_token_id=3025,
        decoder_start_token_id=0,
        feed_forward_proj=t5_config['feed_forward_proj'],
        n_positions=model_config['TIGER']['n_positions'],
    )
    model = T5ForConditionalGeneration(config=model_config).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    if args.cold_start:
        evaluate_cold_start(model, test_dataloader, test_unseen_semantic_ids, device, num_beams=30)
    else:
        evaluate(model, test_dataloader, device, num_beams=30)

if __name__ == "__main__":
    main(parse_args())