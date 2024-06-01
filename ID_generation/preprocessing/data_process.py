from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import tqdm
import statistics
import html
import re
import os
import argparse

def parse(path): # for Amazon
    g = gzip.open(path, 'r')
    for l in g:
        l = l.replace(b'true', b'True').replace(b'false', b'False')
        yield eval(l)

def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    review_mapping = defaultdict(dict)
    # older Amazon
    data_flie = dataset_name
    # latest Amazon
    for inter in parse(data_flie):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        review_mapping[inter['reviewerID']][inter["asin"]] = (inter['summary'], inter['reviewText'])
        datas.append((user, item, int(time)))
    return datas, review_mapping

def Amazon_meta(dataset_name, data_maps):
    '''
    asin - ID of the product, e.g. 0000031852
    --"asin": "0000031852",
    title - name of the product
    --"title": "Girls Ballet Tutu Zebra Hot Pink",
    description
    price - price in US dollars (at time of crawl)
    --"price": 3.17,
    imUrl - url of the product image (str)
    --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
    related - related products (also bought, also viewed, bought together, buy after viewing)
    --"related":{
        "also_bought": ["B00JHONN1S"],
        "also_viewed": ["B002BZX8Z6"],
        "bought_together": ["B002BZX8Z6"]
    },
    salesRank - sales rank information
    --"salesRank": {"Toys & Games": 211836}
    brand - brand name
    --"brand": "Coxlures",
    categories - list of categories the product belongs to
    --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    '''
    datas = {}
    meta_file = './ID_generation/preprocessing/raw_data/meta_' + dataset_name + '.json.gz'
    item_asins = set(data_maps['item2id'].keys())
    for info in parse(meta_file):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas

def Yelp(date_min, date_max, rating_score):
    datas = []
    data_flie = 'yelp_academic_dataset_review.json'
    lines = open(data_flie).readlines()
    for line in tqdm.tqdm(lines):
        review = json.loads(line.strip())
        user = review['user_id']
        item = review['business_id']
        rating = review['stars']
        # 2004-10-12 10:13:32 2019-12-13 15:51:19
        date = review['date']
        # 剔除一些例子
        if date < date_min or date > date_max or float(rating) <= rating_score:
            continue
        time = date.replace('-','').replace(':','').replace(' ','')
        datas.append((user, item, int(time)))
    return datas


def Yelp_meta(datamaps):
    meta_infos = {}
    meta_file = 'yelp_academic_dataset_business.json'
    item_ids = set(datamaps['item2id'].keys())
    lines = open(meta_file).readlines()
    for line in tqdm.tqdm(lines):
        info = json.loads(line)
        if info['business_id'] not in item_ids:
            continue
        meta_infos[info['business_id']] = info
    return meta_infos

def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items

def get_attribute_Amazon(meta_infos, datamaps, attribute_core):

    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        for cates in info['categories']:
            for cate in cates[1:]: # 把主类删除 没有用
                attributes[cate] +=1
        # To Reproduce TIGER Code distribution experiment, we don't consider brands
        # try:
        #     attributes[info['brand']] += 1
        # except:
        #     pass

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        # try:
        #     if attributes[info['brand']] >= attribute_core:
        #         new_meta[iid].append(info['brand'])
        # except:
        #     pass
        for cates in info['categories']:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def get_attribute_Yelp(meta_infos, datamaps, attribute_core):
    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        try:
            cates = [cate.strip() for cate in info['categories'].split(',')]
            for cate in cates:
                attributes[cate] +=1
        except:
            pass
    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []
        try:
            cates = [cate.strip() for cate in info['categories'].split(',') ]
            for cate in cates:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
        except:
            pass
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'after delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes

def meta_map(meta_infos, data_maps, features_needed=['title', 'price', 'brand', 'feature', 'categories', 'description']):
    id2meta={}
    item2meta={}

    for item,meta in tqdm.tqdm(meta_infos.items()):
        meta_text = ''
        keys = set(meta.keys())
        for feature in features_needed:
            if feature in keys:
                meta_text += feature_process(meta[feature])
        item2meta[item]=meta_text
        id=data_maps['item2id'][item]
        id2meta[id]=meta_text

            # if len(v2)>0 and isinstance(v2[0], list):
            #     meta[i2] = clean_text(v2[0])
            # else:
            #     meta[i2]=clean_text(v2)
        # meta_clean = clean_text(meta)

    return data_maps, id2meta


def get_item_review_map(review_mapping, data_maps, meta_infos):
    id2review = defaultdict(dict)
    for reviewer_id, items in review_mapping.items():
        user_id = data_maps['user2id'][reviewer_id]
        for item, review in items.items():
            id=data_maps['item2id'][item]
            title = "" if 'title' not in meta_infos[item] else meta_infos[item]["title"]
            categories = meta_infos[item]['categories'] if 'categories' in meta_infos[item] else ""
            id2review[user_id][id] = (title, categories,) + review
            
    return id2review


def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def id_map(user_items): # user_items dict

    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps

def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(', '.join(l))
    else:

        return l

def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text

def feature_process(feature):
    sentence=""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += '.'
    elif len(feature) > 0 and isinstance(feature[0], list):
        for v1 in feature:
            for v in v1:
                sentence += clean_text(v)
                sentence += ', '
        sentence = sentence[:-2]
        sentence += '.'
    elif isinstance(feature, list):
        for v1 in feature:
            sentence += clean_text(v1)
    else:
        sentence= clean_text(feature)
    return sentence + ' '

def preprocessing(config, require_attributes = False):
    dataset_name, data_type, features_needed = config["name"], config["type"], config["features_needed"]
    features_used = "_".join(features_needed)
    data_file =  f"./ID_generation/preprocessing/processed/{dataset_name}.txt"
    id2meta_file = f"./ID_generation/preprocessing/processed/{dataset_name}_{features_used}_id2meta.json"
    item2attributes_file = f"./ID_generation/preprocessing/processed/{dataset_name}_item2attributes.json"
    attributesmap_file = f"./ID_generation/preprocessing/processed/{dataset_name}_attributesmap.json"

    # if require_attributes:
    #     if os.path.exists(data_file) and os.path.exists(id2meta_file) and os.path.exists(item2attributes_file) and os.path.exists(attributesmap_file):
    #         print(f'{dataset_name} has been processed!')
    #         return
    # else:
    #     if os.path.exists(data_file) and os.path.exists(id2meta_file):
    #         print(f'{dataset_name} has been processed!')
    #         return
    
    print(f"data_name: {dataset_name}, data_type: {data_type}, require_attributes: {require_attributes}")
        
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    # reviews = {}
    # meta_file = 'reviews_{}_5.json.gz'.format(data_name)
    # for i, info in enumerate(parse(meta_file)):
    #     datas[i] = info
    if data_type=='yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2019-01-01 00:00:00'
        datas = Yelp(date_min, date_max, rating_score)
    else:
        datas, review_mapping = Amazon('./ID_generation/preprocessing/raw_data/reviews_'+dataset_name + '_5.json.gz', rating_score=rating_score)

    user_items = get_interaction(datas)

    print(f'{dataset_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    user_items_id, user_num, item_num, data_maps = id_map(user_items)
    user_count, item_count, _ = check_Kcore(user_items_id, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    seqs_length = [len(user_items_id[i]) for i in user_items_id.keys()]
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%\n'+ \
                f'Sequence Length Mean: {(sum(seqs_length) / len(seqs_length)):.2f}, Mediam: {statistics.median(seqs_length)}'
    print(show_info)

    print('Begin extracting meta infos...')

    if data_type == 'Amazon':
        meta_infos = Amazon_meta(dataset_name, data_maps)
        if require_attributes:
            attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps, attribute_core)
        data_maps, id2meta=meta_map(meta_infos, data_maps, features_needed)
        item2review = get_item_review_map(review_mapping, data_maps, meta_infos)
    else:
        meta_infos = Yelp_meta(data_maps)
        if require_attributes:
            attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Yelp(meta_infos, data_maps, attribute_core)
        data_maps, id2meta = meta_map(meta_infos, data_maps, features_needed)

    if require_attributes:
        print(f'{dataset_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
            f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
            f'{avg_attribute:.1f} \\')
    else:
        print(f'{dataset_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
            f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\% \\')
        
    # -------------- Save Data ---------------
    
    os.makedirs("./ID_generation/preprocessing/processed/", exist_ok=True)
    
    with open(data_file, 'w') as out:
        for user, items in user_items_id.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    json_str = json.dumps(id2meta)
    with open(id2meta_file, 'w') as out:
        out.write(json_str)

    os.makedirs('./dataset', exist_ok=True)
    json_str = json.dumps(item2review)
    with open(f'./dataset/item2review_{dataset_name}.json', 'w') as out:
        out.write(json_str)

    # id2sen_file = data_name + '_id2sentence.json'
    # json_str = json.dumps(id2sentence)
    # with open(id2sen_file, 'w') as out:
    #     out.write(json_str)
    
    if require_attributes:
        json_str = json.dumps(item2attributes)
        with open(item2attributes_file, 'w') as out:
            out.write(json_str)

        json_str = json.dumps(datamaps)
        with open(attributesmap_file, 'w') as out:
            out.write(json_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--dataset_name", type=str, required=True, nargs='+', help="The names of the datasets")
    parser.add_argument("-t", "--data_type", type=str, choices=['Amazon', 'yelp'], default='Amazon', help="The type of the data (Amazon or yelp)")
    parser.add_argument("-a", "--require_attributes", action="store_true", help="If set, require to extract attributes mappings")
    args = parser.parse_args()

    kwargs = {}

    if args.data_type is not None:
        kwargs["data_type"] = args.data_type
    if args.require_attributes:
        kwargs["require_attributes"] = args.require_attributes

    for name in args.dataset_name:
        kwargs['dataset_name'] = name
        preprocessing(**kwargs)