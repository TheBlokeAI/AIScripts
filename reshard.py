#
# A simple script for re-sharding a model to the specified size
# And also saving it as float16 (or bfloat16) if it's currently in float32
#

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_shard_size", type=str, default="9GiB")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")

    return parser.parse_args()

def main():
    args = get_args()

    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }

    if args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    elif args.dtype == 'float32':
        dtype = torch.bfloat32

    print(f"Loading base model: {args.base_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        **device_arg
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    model.save_pretrained(args.output_dir, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(f"{args.output_dir}")
    print(f"Model saved to {args.output_dir} with max_shard_size={args.max_shard_size}")

if __name__ == "__main__" :
    main()
