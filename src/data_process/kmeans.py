
import os
import torch
import torch.nn.functional as F
from lib.HF_Model import HFTransformerModel as TransformerModel
from lib.Tree_Model import Tree
import numpy as np
import json
import tqdm


data_din_path='PATH/dataset.emb-inter-DIN_MODEL-td.npy'
output_din_file='PATH/dataset.din.index.json'
data_text_path='PATH/dataset.emb-llama-td.npy'
output_text_file='PATH/dataset.text.index.json'

output_file = 'PATH/dataset.concat.index.json'

data = torch.from_numpy(np.load(data_din_path)).cuda()
max_iters=100
feature_ratio=1.0
item_num=data.shape[0]
k = 256
init_way='embkm'
parall=20


tree = Tree(data=data,max_iters=max_iters,feature_ratio=feature_ratio,\
                                item_num=item_num,k=k,init_way=init_way,parall=parall)


item_to_code_mat=torch.full((item_num,tree.tree_height),-1,dtype=torch.int64)
for item_id,paths in tree.item_to_code.items():
    assert len(paths)>0
    item_to_code_mat[item_id]=paths[0]
item_to_code_mat


all_indices = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in range(item_to_code_mat.size(0)):
    codes = item_to_code_mat[d]
    code = []
    for i, ind in enumerate(codes):
        code.append(prefix[i].format(int(ind)))
    all_indices.append(code)
all_indices

all_indices_dict = {}
for item, indices in enumerate(all_indices):
    all_indices_dict[item] = list(indices)

with open(output_din_file, 'w') as fp:
    json.dump(all_indices_dict,fp)


data = torch.from_numpy(np.load(data_text_path)).cuda()
max_iters=100
feature_ratio=1.0
item_num=data.shape[0]
print('item_num:',item_num)
k = 256
init_way='embkm'
parall=20

tree = Tree(data=data,max_iters=max_iters,feature_ratio=feature_ratio,\
                                item_num=item_num,k=k,init_way=init_way,parall=parall)

item_to_code_mat=torch.full((item_num,tree.tree_height),-1,dtype=torch.int64)
for item_id,paths in tree.item_to_code.items():
    assert len(paths)>0
    item_to_code_mat[item_id]=paths[0]
item_to_code_mat

all_indices = []

for d in range(item_to_code_mat.size(0)):
    codes = item_to_code_mat[d]
    code = []
    for i, ind in enumerate(codes):
        code.append(prefix[i].format(int(ind)))
    all_indices.append(code)
all_indices

all_indices_dict = {}
for item, indices in enumerate(all_indices):
    all_indices_dict[item] = list(indices)

with open(output_text_file, 'w') as fp:
    json.dump(all_indices_dict,fp)


def merge(indices_text, indices_din):

    indices = {}
    assert len(indices_text) == len(indices_din)
    for i in indices_text.keys():
        indices[i] = indices_text[i] + indices_din[i]
        for j in range(len(indices[i])):
            value_list = list(indices[i][j])
            value_list[1] = chr(ord('a') + j)
            indices[i][j] = ''.join(value_list)

    return indices

indices_text = json.load(output_text_file)

indices_din = json.load(output_din_file)

indices = merge(indices_text, indices_din)


json.dump(indices,output_file)