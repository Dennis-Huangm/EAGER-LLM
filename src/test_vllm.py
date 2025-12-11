import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams

from utils import set_seed, parse_global_args, parse_dataset_args, parse_test_args, load_test_dataset
from evaluate import get_topk_results, get_metrics_results


def test_vllm(args):
    """
    Test using vLLM for high-throughput inference.
    vLLM provides significant speedup through:
    - PagedAttention for efficient KV cache management
    - Continuous batching for better GPU utilization
    - Optimized CUDA kernels
    """
    set_seed(args.seed)
    print(vars(args))

    # Load test dataset
    test_data = load_test_dataset(args)
    all_items = test_data.get_all_items_without_dup()
    print(f"Data num: {len(test_data)}")

    # Prepare all prompts
    print("Preparing prompts...")
    prompts = []
    targets = []
    for i in range(len(test_data)):
        sample = test_data[i]
        prompts.append(sample["input_ids"])
        targets.append(sample["labels"])

    # Initialize vLLM engine
    print(f"Loading model from {args.ckpt_path}...")
    
    # Configure tensor parallelism based on available GPUs
    num_gpus = torch.cuda.device_count()
    tensor_parallel_size = min(num_gpus, args.tensor_parallel_size) if hasattr(args, 'tensor_parallel_size') else 1
    
    llm = LLM(
        model=args.ckpt_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization if hasattr(args, 'gpu_memory_utilization') else 0.9,
        max_model_len=args.max_model_len if hasattr(args, 'max_model_len') else 2048,
        max_logprobs=args.num_beams * 2,  # beam_search needs logprobs >= beam_width
    )

    # Configure beam search parameters for vLLM v0.12+
    beam_search_params = BeamSearchParams(
        beam_width=args.num_beams,
        max_tokens=6,  # One SID (max 5 tokens) + buffer
        temperature=0.0,
    )

    # Process in batches for memory efficiency
    batch_size = args.test_batch_size
    metrics = args.metrics.split(",")
    wrong_code = 0
    total_code = 0
    metrics_results = {}
    total = 0

    print("Starting evaluation...")
    
    # Process all prompts in batches
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        
        batch_prompts = prompts[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]
        bs = len(batch_targets)
        
        # Convert prompts to dict format required by beam_search
        # beam_search expects List[Dict] with "prompt" key
        batch_prompts_dict = [{"prompt": p} for p in batch_prompts]
        
        # Generate with vLLM beam search
        # beam_search returns List[BeamSearchOutput] where each BeamSearchOutput has sequences
        outputs = llm.beam_search(batch_prompts_dict, beam_search_params)
        # Process outputs
        # Note: seq.text contains the FULL sequence (prompt + generated), need to remove prompt
        predictions = []
        scores = []
        
        for i, output in enumerate(outputs):
            prompt_text = batch_prompts[i]  # Original prompt for this sample
            # BeamSearchOutput contains sequences (list of BeamSearchSequence)
            # Each BeamSearchSequence has: text, cum_logprob, finish_reason
            for seq in output.sequences:
                # Remove the prompt prefix from the full text to get only generated part
                full_text = seq.text
                if full_text.startswith(prompt_text):
                    generated_text = full_text[len(prompt_text):]
                else:
                    generated_text = full_text
                pred_text = generated_text.strip().replace(" ", "").rstrip(",")
                predictions.append(pred_text)
                scores.append(seq.cum_logprob if seq.cum_logprob is not None else 0.0)
        
        # Convert scores to tensor for compatibility with get_topk_results
        scores = torch.tensor(scores)
        
        # Get topk results
        topk_res, wrong_code_single = get_topk_results(
            predictions, scores, batch_targets, args.num_beams,
            all_items=all_items if args.filter_items else None
        )
        wrong_code += wrong_code_single
        total_code += len(predictions)
        total += bs

        # Calculate metrics
        batch_metrics_res = get_metrics_results(topk_res, metrics)
        for m, res in batch_metrics_res.items():
            if m not in metrics_results:
                metrics_results[m] = res
            else:
                metrics_results[m] += res

        # Print intermediate results
        if (batch_idx + 1) % 50 == 0:
            temp = {}
            for m in metrics_results:
                temp[m] = metrics_results[m] / total
            print(temp)
            print(f'{wrong_code} / {total_code} = {wrong_code/total_code:.4f}')

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
    
    results_dir = os.path.dirname(args.results_file)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)
    print("Results saved to: ", args.results_file)


def parse_vllm_args(parser):
    """Add vLLM-specific arguments"""
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization ratio (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=2048,
                        help="Maximum model context length")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test_vllm")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser = parse_vllm_args(parser)

    args = parser.parse_args()

    test_vllm(args)
