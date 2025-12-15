import argparse
import json
import os
import sys

import torch
import transformers
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results


def test_ddp(args):

    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank, device_id=torch.device("cuda", local_rank))

    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # Required for batch generation with decoder-only models
    if args.lora:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
        )
    # assert model.config.vocab_size == len(tokenizer)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True, shuffle=False)

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items_without_dup()

    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)


    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             sampler=ddp_sampler, num_workers=2, pin_memory=True)

    if local_rank == 0:
        print("data num:", len(test_data))

    model.eval()

    metrics = args.metrics.split(",")
    wrong_code = 0
    total_code = 0
    metrics_results = {}
    total = 0
    
    if local_rank == 0:
        print("Starting evaluation...")
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            inputs = batch[0].to(device)
            print(inputs["input_ids"][0])
            targets = batch[1]
            print(targets[0])
            bs = len(targets)
            num_beams = args.num_beams
            while True:
                try:
                    generate_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "max_new_tokens": 6,  # One SID (max 5 tokens) + buffer
                        "num_beams": num_beams,
                        "num_return_sequences": num_beams,
                        "output_scores": True,
                        "return_dict_in_generate": True,
                        "early_stopping": True,
                    }
                    if prefix_allowed_tokens is not None:
                        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens
                    
                    output = model.module.generate(**generate_kwargs)
                    break
                except torch.cuda.OutOfMemoryError as e:
                    print("Out of memory!")
                    num_beams = num_beams - 1
                    print("Beam:", num_beams)
                except Exception:
                    raise RuntimeError

            output_ids = output["sequences"]
            scores = output["sequences_scores"]
            
            # Extract only newly generated tokens (remove input part)
            input_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_length:]
            
            # Decode only the generated part to get predicted SIDs
            predictions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            # Clean predictions: remove spaces and trailing commas
            predictions = [pred.strip().replace(" ", "").rstrip(",") for pred in predictions]
            print(predictions[:20])

            topk_res, wrong_code_single = get_topk_results(predictions, scores, targets, num_beams,
                                        all_items=all_items if args.filter_items else None)
            wrong_code = wrong_code_single + wrong_code
            total_code = total_code + len(predictions)

            bs_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=bs, object_list=bs_gather_list)
            total += sum(bs_gather_list)
            res_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

            if local_rank == 0:
                all_device_topk_res = []
                for ga_res in res_gather_list:
                    all_device_topk_res += ga_res
                batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if (step + 1) % 10 == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)
                    print(f'{wrong_code} / {total_code} = {wrong_code/total_code}')

            dist.barrier()

    dist.barrier()

    if local_rank == 0:
        # Calculate final metrics
        for m in metrics_results:
            metrics_results[m] = metrics_results[m] / total

        print("======================================================")
        print("Final results: ", metrics_results)
        print(f"Wrong codes: {wrong_code} / {total_code} = {wrong_code/total_code if total_code > 0 else 0:.4f}")
        print("======================================================")

        # Save results
        save_data = {
            "metrics_results": metrics_results,
            "wrong_code": wrong_code,
            "total_code": total_code,
            "error_rate": wrong_code / total_code if total_code > 0 else 0
        }
        
        if not os.path.exists(args.results_file):
            os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)
        print("Results saved to: ", args.results_file)

    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test_ddp(args)
