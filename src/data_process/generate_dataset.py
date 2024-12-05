

import argparse
import os
import os.path as osp
import random
import time
from logging import getLogger
import openai
from utils import load_json, intention_prompt_1, intention_prompt_2, preference_prompt_1, preference_prompt_2, amazon18_dataset2fullname, write_json_file, preference_prompt_2_1, preference_prompt_2_2
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_res_batch(prompt_list, max_tokens):

    token_lists=tokenizer(prompt_list, return_tensors="pt", padding=True)
    token_lists = {key: value.to(device) for key, value in token_lists.items()}
    generated_ids = model.generate(
        input_ids=token_lists["input_ids"],
        attention_mask=token_lists["attention_mask"],
        max_new_tokens=max_tokens, 
        do_sample=True)

    # decode with mistral tokenizer
    result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    result = [r[r.find('[/INST]') + len('[/INST]'):].replace('</s>', '').strip() if '[/INST]' in r else r for r in result]

    return result


def get_intention_train(args, inters, item2feature, reviews):

    intention_train_output_file = os.path.join(args.root,"intention_train.json")


    # Suggest modifying the prompt based on different datasets
    prompts=[]
    prompts.append('<s>[INST]'+intention_prompt_1+'[/INST]')
    prompts.append('<s>[INST]'+intention_prompt_2+'[/INST]')
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()

    prompt_list = [[],[]]

    inter_data = []

    for (user,item_list) in inters.items():
        user = int(user)
        item = int(item_list[-3])
        history = item_list[:-3]

        inter_data.append((user,item,history))

        review = reviews[str((user, item))]["review"]
        item_title = item2feature[str(item)]["title"]
        for i in range(2):
            prompt_list[i].append(prompts[i].format(item_title=item_title,dataset_full_name=dataset_full_name,review=review))
    
    st = 0
    with open(intention_train_output_file, mode='a') as f:
        with tqdm(total=len(prompt_list[0])) as pbar:
            while st < len(prompt_list[0]):

                res_1 = get_res_batch(prompt_list[0][st:st+args.batchsize], args.max_tokens)
                res_2 = get_res_batch(prompt_list[1][st:st+args.batchsize], args.max_tokens)

                for i, answer in enumerate(zip(res_1,res_2)):
                    user, item, history = inter_data[st+i]
                    # print(answer)
                    # print("=============")

                    # if answer == '':
                    #     print("answer null error")
                    #     answer = "I enjoy high-quality item."

                    # if answer.strip().count('\n') != 1:
                    #     if 'haracteristics:' in answer:
                    #         answer = answer.strip().split("The item's characteristics:")
                    #     else:
                    #         answer = answer.strip().split("The item's characteristic:")
                    # else:
                    #     answer = answer.strip().split('\n')

                    # if '' in answer:
                    #     answer.remove('')

                    if len(answer) == 1:
                        print(answer)
                        user_preference = item_character = answer[0]
                    elif len(answer) >= 3:
                        print(answer)
                        answer = answer[-1]
                        user_preference = item_character = answer
                    else:
                        user_preference, item_character = answer

                    if ':' in user_preference:
                        idx = user_preference.index(':')
                        user_preference = user_preference[idx+1:]
                    user_preference = user_preference.strip().replace('}','')
                    user_preference = user_preference.replace('\n','')

                    if ':' in item_character:
                        idx = item_character.index(':')
                        item_character = item_character[idx+1:]
                    item_character = item_character.strip().replace('}','')
                    item_character = item_character.replace('\n','')

                    dict = {"user":user, "item":item, "inters": history,
                            "user_related_intention":user_preference, "item_related_intention": item_character}

                    json.dump(dict, f)
                    f.write("\n")
                pbar.update(args.batchsize)
                st += args.batchsize


    return intention_train_output_file

def get_intention_test(args, inters, item2feature, reviews):

    intention_test_output_file = os.path.join(args.root,"intention_test.json")


    # Suggest modifying the prompt based on different datasets
    prompts=[]
    prompts.append('<s>[INST]'+intention_prompt_1+'[/INST]')
    prompts.append('<s>[INST]'+intention_prompt_2+'[/INST]')
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()

    prompt_list = [[],[]]

    inter_data = []

    for (user,item_list) in inters.items():
        user = int(user)
        item = int(item_list[-3])
        history = item_list[:-3]

        inter_data.append((user,item,history))

        review = reviews[str((user, item))]["review"]
        item_title = item2feature[str(item)]["title"]
        for i in range(2):
            prompt_list[i].append(prompts[i].format(item_title=item_title,dataset_full_name=dataset_full_name,review=review))
    
    st = 0
    with open(intention_test_output_file, mode='a') as f:
        with tqdm(total=len(prompt_list[0])) as pbar:
            while st < len(prompt_list[0]):

                res_1 = get_res_batch(prompt_list[0][st:st+args.batchsize], args.max_tokens)
                res_2 = get_res_batch(prompt_list[1][st:st+args.batchsize], args.max_tokens)

                for i, answer in enumerate(zip(res_1,res_2)):
                    user, item, history = inter_data[st+i]
                    # print(answer)
                    # print("=============")

                    # if answer == '':
                    #     print("answer null error")
                    #     answer = "I enjoy high-quality item."

                    # if answer.strip().count('\n') != 1:
                    #     if 'haracteristics:' in answer:
                    #         answer = answer.strip().split("The item's characteristics:")
                    #     else:
                    #         answer = answer.strip().split("The item's characteristic:")
                    # else:
                    #     answer = answer.strip().split('\n')

                    # if '' in answer:
                    #     answer.remove('')

                    if len(answer) == 1:
                        print(answer)
                        user_preference = item_character = answer[0]
                    elif len(answer) >= 3:
                        print(answer)
                        answer = answer[-1]
                        user_preference = item_character = answer
                    else:
                        user_preference, item_character = answer

                    if ':' in user_preference:
                        idx = user_preference.index(':')
                        user_preference = user_preference[idx+1:]
                    user_preference = user_preference.strip().replace('}','')
                    user_preference = user_preference.replace('\n','')

                    if ':' in item_character:
                        idx = item_character.index(':')
                        item_character = item_character[idx+1:]
                    item_character = item_character.strip().replace('}','')
                    item_character = item_character.replace('\n','')

                    dict = {"user":user, "item":item, "inters": history,
                            "user_related_intention":user_preference, "item_related_intention": item_character}

                    json.dump(dict, f)
                    f.write("\n")
                pbar.update(args.batchsize)
                st += args.batchsize


    return intention_test_output_file


def get_user_preference(args, inters, item2feature, reviews):

    preference_output_file = os.path.join(args.root,"user_preference.json")


    # Suggest modifying the prompt based on different datasets\
    prompt_1 = ('<s>[INST]'+preference_prompt_1+'[/INST]')
    prompt_2 = ('<s>[INST]'+preference_prompt_2_1+'[/INST]')
    prompt_3 = ('<s>[INST]'+preference_prompt_2_2+'[/INST]')


    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    print(dataset_full_name)

    prompt_list_1 = []
    prompt_list_2 = []
    prompt_list_3 = []

    users = []

    for (user,item_list) in inters.items():
        # 14336 17011 19686 22363
        users.append(user)
        history = item_list[:-3]
        item_titles = []
        for j, item in enumerate(history):
            item_titles.append(str(j+1) + '.' + item2feature[str(item)]["title"])
        if len(item_titles) > args.max_his_len:
            item_titles = item_titles[-args.max_his_len:]
        item_titles = ", ".join(item_titles)
        
        input_prompt_1 = prompt_1.format(dataset_full_name=dataset_full_name, item_titles=item_titles)
        input_prompt_2 = prompt_2.format(dataset_full_name=dataset_full_name, item_titles=item_titles)
        input_prompt_3 = prompt_3.format(dataset_full_name=dataset_full_name, item_titles=item_titles)

        prompt_list_1.append(input_prompt_1)
        prompt_list_2.append(input_prompt_2)
        prompt_list_3.append(input_prompt_3)


    st = 0
    with open(preference_output_file, mode='a') as f:
        with tqdm(total=len(prompt_list_1)) as pbar:
            while st < len(prompt_list_1):

                res_1 = get_res_batch(prompt_list_1[st:st + args.batchsize], args.max_tokens)
                res_2 = get_res_batch(prompt_list_2[st:st + args.batchsize], args.max_tokens)
                res_3 = get_res_batch(prompt_list_2[st:st + args.batchsize], args.max_tokens)
                for i, answers in enumerate(zip(res_1, res_2, res_3)):
                    
                    user = users[st + i]

                    answer_1, answer_2, answer_3 = answers

                    if answer_1 == '':
                        print("answer null error")
                        answer_1 = "I enjoy high-quality item."
                        
                    if answer_2 == '':
                        print("answer null error")
                        answer_2 = "I enjoy high-quality item."

                    if answer_3 == '':
                        print("answer null error")
                        answer_3 = "I enjoy high-quality item."

                    short_preference=answer_3
                    long_preference=answer_2

                    if ':' in long_preference:
                        idx = long_preference.index(':')
                        long_preference = long_preference[idx+1:]
                    long_preference = long_preference.strip().replace('}','')
                    long_preference = long_preference.replace('\n','')

                    if ':' in short_preference:
                        idx = short_preference.index(':')
                        short_preference = short_preference[idx+1:]
                    short_preference = short_preference.strip().replace('}','')
                    short_preference = short_preference.replace('\n','')

                    dict = {"user":user,"user_preference":[answer_1, long_preference, short_preference]}
                    # print(dict)
                    json.dump(dict, f)
                    f.write("\n")
                pbar.update(args.batchsize)
                st += args.batchsize

    return preference_output_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Beauty / Sports')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--max_his_len', type=int, default=20)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = 'cuda:0'

    args.root = os.path.join(args.root, args.dataset)

    inter_path = os.path.join(args.root, f'{args.dataset}.inter.json')
    inters = load_json(inter_path)


    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    reviews_path = os.path.join(args.root, f'{args.dataset}.review.json')
    reviews = load_json(reviews_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token    
    tokenizer.padding_side = "left"# 设置填充方向为左边

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=device) #, device_map='auto'
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    preference_output_file = get_user_preference(args, inters, item2feature, reviews)
    intention_train_output_file = get_intention_train(args, inters, item2feature, reviews)
    intention_test_output_file = get_intention_test(args, inters, item2feature, reviews)
    # intention_train_output_file = os.path.join(args.root,"intention_train.json")
    # intention_test_output_file = os.path.join(args.root,"intention_test.json")

    intention_train = {}
    intention_test = {}
    user_preference = {}

    with open(intention_train_output_file, "r") as f:
        for line in f:
            # print(line)
            content = json.loads(line)
            if content["user"] not in intention_train:
                intention_train[content["user"]] = {"item":content["item"],
                                                "inters":content["inters"],
                                                "querys":[ content["user_related_intention"], content["item_related_intention"] ]}


    with open(intention_test_output_file, "r") as f:
        for line in f:
            content = json.loads(line)
            if content["user"] not in intention_train:
                intention_test[content["user"]] = {"item":content["item"],
                                                "inters":content["inters"],
                                                "querys":[ content["user_related_intention"], content["item_related_intention"] ]}


    with open(preference_output_file, "r") as f:
        x = 0
        for line in f:
            content = json.loads(line)
            user_preference[content["user"]] = content["user_preference"]

    user_dict = {
        "user_explicit_preference": user_preference,
        "user_vague_intention": {"train": intention_train, "test": intention_test},
    }

    write_json_file(user_dict, os.path.join(args.root, f'{args.dataset}.Mistral.user.json'))
