import math
import random
from pandas import DataFrame
from lib.generate_training_batches import Train_instance
from lib import DINTrain
import numpy as np
from lib import generate_train_and_test_data
from lib.generate_train_and_test_data import _gen_train_sample, _read, _gen_test_sample
import os
import torch
import sys
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
from tqdm import tqdm
# from utils import *
import torch.nn as nn
import time
sys.path.append('../..')

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def presision(result_list, gt_list, top_k):
    count = 0.0
    for r, g in zip(result_list, gt_list):
        count += len(set(r).intersection(set(g)))
    return count/(top_k*len(result_list))


def recall(result_list, gt_list):
    t = 0.0
    for r, g in zip(result_list, gt_list):
        t += 1.0*len(set(r).intersection(set(g)))/len(g)
    return t/len(result_list)


def f_measure(result_list, gt_list, top_k, eps=1.0e-9):
    f = 0.0
    for r, g in zip(result_list, gt_list):
        recc = 1.0*len(set(r).intersection(set(g)))/len(g)
        pres = 1.0*len(set(r).intersection(set(g)))/top_k
        if recc+pres < eps:
            continue
        f += (2*recc*pres)/(recc+pres)
    return f/len(result_list)


def novelty(result_list, s_u, top_k):
    count = 0.0
    for r, g in zip(result_list, s_u):
        count += len(set(r)-set(g))
    return count/(top_k*len(result_list))


def hit_ratio(result_list, gt_list):
    intersetct_set = [len(set(r) & set(g))
                      for r, g in zip(result_list, gt_list)]
    return 1.0*sum(intersetct_set)/sum([len(gts) for gts in gt_list])


def NDCG_bug(result_list, gt_list):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = set(gt)
        indicator = np.asfarray([1 if r in setgt else 0 for r in re])
        sorted_indicator = indicator[indicator.argsort(-1)[::-1]]
        if 1 in indicator:
            t += np.sum(indicator / np.log2(1.0*np.arange(2, len(indicator) + 2))) /\
                np.sum(sorted_indicator/np.log2(1.0 *
                       np.arange(2, len(indicator) + 2)))
    return t/len(gt_list)


def NDCG_recforest(result_list, gt_list):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = set(gt)
        indicator = np.asfarray([1 if r in setgt else 0 for r in re])
        sorted_indicator = np.ones(min(len(setgt), len(re)))
        if 1 in indicator:
            t += np.sum(indicator / np.log2(1.0*np.arange(2, len(indicator) + 2))) /\
                np.sum(sorted_indicator/np.log2(1.0 *
                       np.arange(2, len(sorted_indicator) + 2)))
    return t/len(gt_list)


def NDCG_comicrec(result_list, gt_list):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        recall = 0
        dcg = 0.0
        setgt = set(gt)
        for no, iid in enumerate(re):
            if iid in setgt:
                recall += 1
                dcg += 1.0 / math.log(no + 2, 2)
        idcg = 0.0
        for no in range(recall):
            idcg += 1.0 / math.log(no + 2, 2)
        if recall > 0:
            t += dcg / idcg
    return t/len(gt_list)


def MAP(result_list, gt_list, topk):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = set(gt)
        indicator = np.asfarray([1 if r in setgt else 0 for r in re])
        t += np.mean([indicator[:i].sum(-1) /
                     i for i in range(1, topk+1)], axis=-1)
    return t/len(gt_list)


def NDCG(result_list, gt_list):
    t = 0.0
    for re, gt in zip(result_list, gt_list):
        setgt = gt
        indicator = np.asfarray([1 if r in setgt else 0 for r in re])
        # np.ones(min(len(setgt), len(re)))
        sorted_indicator = np.ones(len(gt))
        if 1 in indicator:
            t += np.sum(indicator / np.log2(1.0*np.arange(2, len(indicator) + 2))) /\
                np.sum(sorted_indicator/np.log2(1.0 *
                       np.arange(2, len(sorted_indicator) + 2)))
    return t/len(gt_list)


def get_training_data(inter_feature, max_len=20):
    training_data = []
    training_labels = []
    for _, items in inter_feature.items():
        items = items[:-1]
        tmp = items[-max_len:]
        items = items[:-1]
        if len(tmp) < max_len:
            tmp = [item_num] * (max_len - len(tmp)) + tmp
        assert (len(tmp) == max_len)
        # print(len(tmp))
        training_labels.append(tmp[-1])
        training_data.append(tmp[:-1])
        # training_data.append(([-1] * (max_len - len(items[-max_len:-1])) + items[-max_len:-1]) if len(items[-max_len:-1]) < max_len else items[-max_len:-1])
        while (len(items) >= max_len):
            tmp = items[-max_len:]
            items = items[:-1]
            if len(tmp) < max_len:
                tmp = [item_num] * (max_len - len(tmp)) + tmp
            assert (len(tmp) == max_len)
            # print(len(tmp))
            training_labels.append(tmp[-1])
            training_data.append(tmp[:-1])
    return torch.tensor(training_data, dtype=torch.long), torch.tensor(training_labels, dtype=torch.long)

def get_testing_data(inter_feature, max_len=20, sample_size=0):
    testing_data = []
    testing_labels = []
    print('total test:', len(inter_feature))
    # Randomly sample 1000 items from inter_feature
    if(sample_size > 0):
        assert(sample_size <= len(inter_feature) and sample_size > 100)
        sampled_items = random.sample(list(inter_feature.items()), sample_size)
    else:
        sampled_items = inter_feature.items()
    for _, items in sampled_items:
        # items = items[:-1]
        tmp = items[-max_len:]
        items = items[:-1]
        if len(tmp) < max_len:
            tmp = [item_num] * (max_len - len(tmp)) + tmp
        assert (len(tmp) == max_len)
        # print(len(tmp))
        testing_labels.append(tmp[-1])
        testing_data.append(tmp[:-1])
    return torch.tensor(testing_data, dtype=torch.long), torch.tensor(testing_labels, dtype=torch.long).unsqueeze(1)

def get_validing_data(inter_feature, max_len=20):
    testing_data = []
    testing_labels = []
    for _, items in inter_feature.items():
        # items = items[:-1]
        tmp = items[-max_len:]
        items = items[:-1]
        if len(tmp) < max_len:
            tmp = [item_num] * (max_len - len(tmp)) + tmp
        assert (len(tmp) == max_len)
        # print(len(tmp))
        testing_labels.append(tmp[-1])
        testing_data.append(tmp[:-1])
    return torch.tensor(testing_data, dtype=torch.long), torch.tensor(testing_labels, dtype=torch.long).unsqueeze(1)


def iterate_minibatches(*tensors, batch_size=4096, shuffle=True, cycle=True, **kw):
    global is_cycle
    while True:
        yield from DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)
        print('cycle')
        is_cycle = True
        # break
        if not cycle:
            break


def generate_training_records(training_instances, training_labels, batch_size=1024, shuffle=True):
    for batch_x, batch_y in iterate_minibatches(training_instances, training_labels, batch_size=batch_size, shuffle=shuffle):
        yield batch_x, batch_y


def optimizer(params): return torch.optim.Adam(params, lr, amsgrad=True)


######################################## 加载参数##################################
# parametres
# 'gowalla' 'MIND' 'Amazon_All_Beauty' 'Sports_and_Outdoors' 'Toys_and_Games'
is_cycle=False
cycle_Samsara=10#训练多少个epoch测试一次
data_set_name = 'Sports' # Beauty Sports Instruments
device = 'cuda:3'
root = "./data"
batch_number = 1000000
sample_negative_num = 60  # 60
max_len = 20
emb_dim = 1024
topk=5
lr=1e-4
sum_pooling = False
batch_size = 256
feature_groups = [5, 4, 2, 2, 1, 1, 1, 1, 1, 1]
device = torch.device(device)
from_ckpt=False
start_time=time.time()
path = f'./data/{data_set_name}/din_model_-1_{emb_dim}'
log_path = f'{path}/{data_set_name}.log'
os.makedirs(path, exist_ok=True)
os.makedirs(os.path.join(path, 'models'), exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.info(f'emb_dim = {emb_dim}, lr = {lr}, batch_size = {batch_size}')
inter_path = os.path.join(root, data_set_name, f'{data_set_name}.inter.json')
inter_feature = load_json(inter_path)
item_path = os.path.join(root, data_set_name, f'{data_set_name}.item.json')
item_feature = load_json(item_path)
item_num = len(item_feature)
training_data, training_labels = get_training_data(inter_feature, max_len)
print(training_labels.size())
test_data_path = f'{path}/testing_data.pt'
test_labels_path = f'{path}/testing_labels.pt'
if os.path.exists(test_data_path) and os.path.exists(test_labels_path):
    testing_data = torch.load(test_data_path)
    testing_labels = torch.load(test_labels_path)
else:
    testing_data, testing_labels = get_testing_data(inter_feature, max_len, 5000)
    torch.save(testing_data, test_data_path)
    torch.save(testing_labels, test_labels_path)
train_model = DINTrain(item_num=item_num,
                       sample_negative_num=sample_negative_num,
                       emb_dim=emb_dim,
                       device=device,
                       sum_pooling=sum_pooling,
                       feature_groups=feature_groups,
                       optimizer=optimizer)
# train_model.DINModel = nn.DataParallel(train_model.DINModel, device_ids=[0,1])
cycle_num=0
if(from_ckpt == True):
    train_model.DINModel = torch.load(ckpt)
    train_model.batch_num = int(ckpt.split('_')[-1].split('.')[0])
    cycle_num=train_model.batch_num

train_model.DINModel.train()

loss_history, dev_precision_history, dev_recall_history, dev_f_measure_history, dev_novelty_history, dev_ndcg_history, policy_acc = [], [], [], [], [], [], []
test_precision_history, test_recall_history, test_f_measure_history, test_novelty_history, test_ndcg_history = [], [], [], [], []
total_precision_history, total_recall_history, total_f_measure_history, total_novelty_history, total_ndcg_history, total_hit_history = [], [], [], [], [], []


for (batch_x, batch_y) in generate_training_records(training_data, training_labels, batch_size):
    loss = train_model.update_DIN(batch_x, batch_y)
    loss_history.append(loss.item())

    if train_model.batch_num % 10 == 0:
        logger.info("step=%i, mean_loss=%.3f, time=%.3f" %
                    (train_model.batch_num, np.mean(loss_history[-10:]), (time.time() - start_time)))
        start_time = time.time()
        
    if is_cycle == True:
        DIN_Model_path = f'{path}/models/DIN_MODEL_{(cycle_num)}.pt'
        torch.save(train_model.DINModel, DIN_Model_path)
        logger.info('DIN_MODEL_{}saved'.format(cycle_num))
        is_cycle = False

        if (cycle_num) % cycle_Samsara == 0 and cycle_num > 1:
            logger.info('evaluating DIN_MODEL_{}'.format(cycle_num))
            loss_history, dev_precision_history, dev_recall_history, dev_f_measure_history, dev_novelty_history, dev_ndcg_history, policy_acc = [], [], [], [], [], [], []
            test_precision_history, test_recall_history, test_f_measure_history, test_novelty_history, test_ndcg_history = [], [], [], [], []
            total_precision_history, total_recall_history, total_f_measure_history, total_novelty_history, total_ndcg_history, total_hit_history = [], [], [], [], [], []

            train_model.DINModel.eval()
            gt_history = testing_labels.numpy()

            all_items = torch.arange(item_num, device=device).view(-1, 1)
            preference_matrix = torch.full((len(testing_data), item_num), -1.0e9, dtype=torch.float32)
            batch_size = 1000
            logger.info(testing_data.shape)
            f_num = testing_data.shape[1]
            # logger.info(item_num,test_batch.shape)
            for i, user in tqdm(enumerate(testing_data), total=len(testing_data)):
                if batch_size >= item_num:
                    part_labels = all_items
                    with torch.no_grad():
                            preference_matrix[i] = train_model.calculate_preference( \
                                user.to(device).expand(len(part_labels), f_num), part_labels.to(device)).view(1, -1).cpu()
                else:
                    start_id = 0
                    while start_id < item_num:
                        part_labels = all_items[start_id:start_id + batch_size, :]
                        # logger.info(len(part_labels),)
                        with torch.no_grad():
                            preference_matrix[i, start_id:start_id + batch_size] = train_model.calculate_preference( \
                                user.to(device).expand(len(part_labels), f_num), part_labels.to(device)).view(1, -1).cpu()
                        start_id = start_id + batch_size
            result_history = preference_matrix.argsort(dim=-1)[:, -topk:].numpy()
            result_history = result_history[:, ::-1]
            # total_precision_history.append(presision(resutl_history, gt_history, topk))
            total_recall_history.append(recall(result_history, gt_history))
            # total_f_measure_history.append(f_measure(resutl_history, gt_history, topk))
            # total_novelty_history.append(novelty(resutl_history, testing_data.tolist(), topk))
            total_ndcg_history.append(NDCG(result_history, gt_history))
            total_hit_history.append(hit_ratio(result_history, gt_history))
            # logger.info('precision: {}'.format(total_precision_history[-1]))
            logger.info('recall: {}'.format(total_recall_history[-1]))
            # logger.info('f-score: {}'.format(total_f_measure_history[-1]))
            logger.info('ndcg: {}'.format(total_ndcg_history[-1]))
            logger.info('hit_rate: {}'.format(total_hit_history[-1]))
            logger.info('*************************************************************************************')
            train_model.DINModel.train()
        cycle_num += 1
        
    if train_model.batch_num > batch_number:
        break