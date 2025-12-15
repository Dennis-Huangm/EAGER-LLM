import argparse
import os

import sys
from typing import List

import torch
import transformers

from transformers import AutoTokenizer, AutoConfig

from utils import *
from collator import Collator
from transformers import AutoModelForCausalLM
from torch import nn

def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    # Pure DDP data-parallel: do NOT set device_map, let Trainer/Accelerate handle device placement.
    # DeepSpeed also should not use device_map.
    if (not ddp) and (not args.deepspeed):
        device_map = "auto"
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="left",  # Qwen uses left padding
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gradient_checkpointing = True
    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    data_collator = Collator(args, tokenizer)
    model_load_kwargs = dict(trust_remote_code=True)
    
    if args.bf16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_load_kwargs["torch_dtype"] = torch.float16
    if device_map is not None:
        model_load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_load_kwargs,
    )
    model.resize_token_embeddings(len(tokenizer))

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    # Configure training arguments based on whether validation data exists
    training_args_dict = dict(
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_step,
        optim=args.optim,
        gradient_checkpointing=gradient_checkpointing,
        save_strategy=args.save_and_eval_strategy,
        save_steps=args.save_and_eval_steps,
        output_dir=args.output_dir,
        save_total_limit=5,
        ddp_find_unused_parameters=False if ddp else None,
        report_to=None,
        remove_unused_columns=False,
    )

    if args.deepspeed:
        training_args_dict["deepspeed"] = args.deepspeed
    
    # Only add eval-related args if valid_data exists
    if valid_data is not None:
        training_args_dict.update(
            eval_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            load_best_model_at_end=True,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        )
    else:
        training_args_dict.update(
            eval_strategy="no",
            load_best_model_at_end=False,
        )
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(**training_args_dict),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)