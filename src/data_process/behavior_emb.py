import argparse
import os
import torch
import numpy as np
from utils import *

from lib.DIN_Model import DeepInterestNetwork

def generate_item_embedding(args):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)
    item_path = os.path.join(args.data_path, f'{args.dataset}.item.json')
    item_feature = load_json(item_path)
    item_num = len(item_feature)

    DIN_model_path = args.din_checkpoint
    DIN_Model=DeepInterestNetwork(item_num=item_num)
    
    DIN_Model=torch.load(DIN_model_path, map_location=torch.device(device))
    embeddings = DIN_Model.item_embedding.embed.weight.data[:item_num,:].cpu()

    embeddings = embeddings.numpy()
    print('Embeddings shape: ', embeddings.shape)
    din_name = os.path.basename(DIN_model_path).split('.')[0]
    file = os.path.join(args.data_path,'emb-inter-' + din_name + "-td" + ".npy")
    
    np.save(file, embeddings)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='Instruments / Beauty / Sports')
    parser.add_argument('--din_checkpoint', type=str,default='')
    parser.add_argument('--gpu_id', type=int, default=3, help='ID of running GPU')
    parser.add_argument('--data_path', type=str, default='none')

    return parser.parse_args(args)


if __name__ == '__main__':
    
    args = parse_args()
    device = set_device(args.gpu_id)
    args.device = device

    generate_item_embedding(args)


