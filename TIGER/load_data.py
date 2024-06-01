from collections import defaultdict
import pickle
import numpy as np
import torch
import os

def defaultdict_defaultdict_defaultdict_int():
    def defaultdict_defaultdict_int():
        def defaultdict_int():
            return defaultdict(int)
        return defaultdict(defaultdict_int)
    return defaultdict(defaultdict_defaultdict_int)

#padding:0 first id:1-256 second id:257-512 third id:513-768 fourth id:769-1024 user id:1025-3024 <EOS>:3025
def expand_id(id):
    return id[0]+1, id[1]+257, id[2]+513, id[3]+769

def pad_sequence(sequence, length):
    # Input sequence format: [<user_id>, <item_1_code_1>, <item_1_code_2>, <item_1_code_3>, <item_1_code_4>, <item_2_code_1>, ....]
    # Here, we have to reserve a position for <EOS>
    if len(sequence) + 1 >= length:
        # No need to pad 
        return [sequence[0]] + sequence[len(sequence) - (length-2):len(sequence)] + [3025]
    return sequence + [3025] + [0]*(length-len(sequence) - 1)

def pad_sequence_attention(sequence, length):
    if len(sequence) + 1 >= length:
        # No need to pad 
        return [sequence[0]] + sequence[len(sequence) - (length-2):len(sequence)] + [1]
    return sequence +  [1] + [0]*(length-len(sequence) - 1)

def generate_input_sequence(user_id, user_sequence, item_2_semantic_id, max_sequence_length):
    input_ids = [user_id]
    attention_mask = [1]
    labels = []
    for i in range(len(user_sequence)):
        if i == len(user_sequence) - 1:
            labels.extend(expand_id(item_2_semantic_id[user_sequence[i]]))
        else:
            input_ids.extend(expand_id(item_2_semantic_id[user_sequence[i]]))
            attention_mask.extend([1]*4)
    labels = np.array(labels + [3025])
    input_ids = np.array(pad_sequence(input_ids, max_sequence_length))
    attention_mask = np.array(pad_sequence_attention(attention_mask, max_sequence_length))
    assert not np.any(labels == 0)
    labels[labels == 0] = -100
    return input_ids, attention_mask, labels

def load_data(dataset, path, max_length=258, max_items_per_seq=np.inf):
    semantic_id_2_item = defaultdict(defaultdict_defaultdict_defaultdict_int)
    item_2_semantic_id = {}
    semantic_ids = pickle.load(open(f'{os.path.abspath(".")}/ID_generation/ID/{path}', 'rb'))

    for i in range(len(semantic_ids)):
        id = semantic_ids[i]
        id_dict = semantic_id_2_item[id[0]][id[1]][id[2]]
        id_dict[len(id_dict)] = i+1
        item_2_semantic_id[i+1] = (*id, len(id_dict))

    assert len(item_2_semantic_id) == semantic_ids.shape[0], "Not all semanticid -> item collisions have been avoided!"

    user_sequence = []
    with open(f'{os.path.abspath(".")}/ID_generation/preprocessing/processed/{dataset}.txt', 'r') as f:
        for line in f.readlines():
            user_sequence.append([int(x) for x in line.split(' ')[1:]])
    user_sequence = [seq if len(seq) <= max_items_per_seq+1 else seq[:max_items_per_seq+1] for seq in user_sequence]
    max_sequence_length = max_length

    training_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    val_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    test_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for i in range(len(user_sequence)):
        user_id = 1025 + i%2000
        train_sequence = []
        train_attention_mask = []
        train_label = []
        val_sequence = []
        val_attention_mask = []
        val_label = []
        test_sequence = []
        test_attention_mask = []
        test_label = []
        
        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]
        for j in range(2, len(user_sequence[i])+1):
            input_ids, attention_mask, labels = generate_input_sequence(user_id, user_sequence[i][:j], item_2_semantic_id, max_sequence_length)
            if j == len(user_sequence[i]) - 1:
                val_sequence.append(input_ids)
                val_attention_mask.append(attention_mask)
                val_label.append(labels)
            elif j == len(user_sequence[i]):
                test_sequence.append(input_ids)
                test_attention_mask.append(attention_mask)
                test_label.append(labels)
            else:
                train_sequence.append(input_ids)
                train_attention_mask.append(attention_mask)
                train_label.append(labels)
        
        training_data['input_ids'].extend(train_sequence)
        training_data['attention_mask'].extend(train_attention_mask)
        training_data['labels'].extend(train_label)
        val_data['input_ids'].extend(val_sequence)
        val_data['attention_mask'].extend(val_attention_mask)
        val_data['labels'].extend(val_label)
        test_data['input_ids'].extend(test_sequence)
        test_data['attention_mask'].extend(test_attention_mask)
        test_data['labels'].extend(test_label)

    training_data['input_ids'] = torch.tensor(training_data['input_ids'], dtype=torch.long)
    training_data['attention_mask'] = torch.tensor(training_data['attention_mask'], dtype=torch.long)
    training_data['labels'] = torch.tensor(training_data['labels'], dtype=torch.long)
    val_data['input_ids'] = torch.tensor(val_data['input_ids'], dtype=torch.long)
    val_data['attention_mask'] = torch.tensor(val_data['attention_mask'], dtype=torch.long)
    val_data['labels'] = torch.tensor(val_data['labels'], dtype=torch.long)
    test_data['input_ids'] = torch.tensor(test_data['input_ids'], dtype=torch.long)
    test_data['attention_mask'] = torch.tensor(test_data['attention_mask'], dtype=torch.long)
    test_data['labels'] = torch.tensor(test_data['labels'], dtype=torch.long)
    return training_data, val_data, test_data

def load_data_cold_start(dataset, path, max_length=258, max_items_per_seq=np.inf):
    semantic_id_2_item = defaultdict(defaultdict_defaultdict_defaultdict_int)
    item_2_semantic_id = {}
    semantic_ids = pickle.load(open(f'{os.path.abspath(".")}/ID_generation/ID/{path}', 'rb'))

    for i in range(len(semantic_ids)):
        id = semantic_ids[i]
        id_dict = semantic_id_2_item[id[0]][id[1]][id[2]]
        id_dict[len(id_dict)] = i+1
        item_2_semantic_id[i+1] = (*id, len(id_dict))

    assert len(item_2_semantic_id) == semantic_ids.shape[0], "Not all semanticid -> item collisions have been avoided!"
    # split into train/val/test items
    val_test_idxs = np.random.choice(np.arange(1, len(semantic_ids)+1), size=int(0.1 * len(semantic_ids)), replace=False)
    test_idxs = np.random.choice(val_test_idxs, size=int(0.5 * len(val_test_idxs)), replace=False)
    val_idxs = np.setxor1d(val_test_idxs, test_idxs)
    val_unseen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in val_idxs])
    test_unseen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in test_idxs])

    user_sequence = []
    with open(f'{os.path.abspath(".")}/ID_generation/preprocessing/processed/{dataset}.txt', 'r') as f:
        for line in f.readlines():
            user_sequence.append([int(x) for x in line.split(' ')[1:]])
    user_sequence = [seq if len(seq) <= max_items_per_seq+1 else seq[:max_items_per_seq+1] for seq in user_sequence]
    max_sequence_length = max_length

    training_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    val_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    test_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for i in range(len(user_sequence)):
        user_id = 1025 + i%2000
        train_sequence = []
        train_attention_mask = []
        train_label = []
        val_sequence = []
        val_attention_mask = []
        val_label = []
        test_sequence = []
        test_attention_mask = []
        test_label = []
        
        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]
        for j in range(2, len(user_sequence[i])+1):
            input_ids, attention_mask, labels = generate_input_sequence(user_id, user_sequence[i][:j], item_2_semantic_id, max_sequence_length)
            if user_sequence[i][j-1] in val_idxs:
                val_sequence.append(input_ids)
                val_attention_mask.append(attention_mask)
                val_label.append(labels)
            elif user_sequence[i][j-1] in test_idxs:
                test_sequence.append(input_ids)
                test_attention_mask.append(attention_mask)
                test_label.append(labels)
            else:
                train_sequence.append(input_ids)
                train_attention_mask.append(attention_mask)
                train_label.append(labels)
        
        training_data['input_ids'].extend(train_sequence)
        training_data['attention_mask'].extend(train_attention_mask)
        training_data['labels'].extend(train_label)
        val_data['input_ids'].extend(val_sequence)
        val_data['attention_mask'].extend(val_attention_mask)
        val_data['labels'].extend(val_label)
        test_data['input_ids'].extend(test_sequence)
        test_data['attention_mask'].extend(test_attention_mask)
        test_data['labels'].extend(test_label)

    training_data['input_ids'] = torch.tensor(training_data['input_ids'], dtype=torch.long)
    training_data['attention_mask'] = torch.tensor(training_data['attention_mask'], dtype=torch.long)
    training_data['labels'] = torch.tensor(training_data['labels'], dtype=torch.long)
    val_data['input_ids'] = torch.tensor(val_data['input_ids'], dtype=torch.long)
    val_data['attention_mask'] = torch.tensor(val_data['attention_mask'], dtype=torch.long)
    val_data['labels'] = torch.tensor(val_data['labels'], dtype=torch.long)
    test_data['input_ids'] = torch.tensor(test_data['input_ids'], dtype=torch.long)
    test_data['attention_mask'] = torch.tensor(test_data['attention_mask'], dtype=torch.long)
    test_data['labels'] = torch.tensor(test_data['labels'], dtype=torch.long)
    return training_data, val_data, test_data, val_unseen_semantic_ids, test_unseen_semantic_ids