# 
# A rather complex AutoGPTQ testing/benchmark script
# It can also load and benchmark unquantised Transformers models
# This was never finished and is messy in places.
# This has not been tested or used since AutoGPTQ 0.1.0 - may not work any more
# For AutoGPTQ benchmarking, it's recommended to use the script provided with AutoGPTQ instead: examples/benchmark/generation_speed.py
#

import sys
import argparse
import time
import csv
import random
from pprint import pprint
from os.path import isfile
from contextlib import contextmanager
from tqdm import tqdm
from logging import getLogger
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

logger = getLogger(__name__)

prompt_templates = {
    'alpaca': "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction: %1\n### Response:",
    'pygmalion': "[CHARACTER]'s Persona: A chatty person\n<START>\nYou: %1\n[CHARACTER]:",
    'vicuna-v0': "### Human: %1\n### Assistant:",
    'open-assistant': "<|prompter|>%1 <|assistant|>",
    'none': "%1"
    }

def get_prompt(template, prompt):
    if template in prompt_templates:
        return prompt_templates[template].replace('%1', prompt)
    else:
        raise ValueError(f"prompt template {template} not implemented.")

class LanguageModel:
    def __init__(self, model_dir, device, tokenizer_dir=None, use_fast_tokenizer=True, trust_remote_code=False, **kwargs):
        self.pipe = None
        self.timing = False
        self.device = device
        tokenizer_dir = tokenizer_dir or model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code)
        self.create_model(model_dir, trust_remote_code=trust_remote_code, **kwargs)

    @staticmethod
    @contextmanager
    def do_timing(timing, reason = None):
        result = {'time': None}
        try:
            if timing:
                start_time = time.time()
            yield result
        finally:
            if timing:
                result['time'] = time.time() - start_time
                if reason: logger.info(f"Time to {reason}: {result['time']:.4f}s")

    @property
    def seed(self):
        return self._current_seed

    @seed.setter
    def seed(self, seed):
        self._seed = int(seed)

    def update_seed(self):
        self._current_seed = (self._seed == -1 ) and random.randint(1, 2**31) or self._seed
        random.seed(self._current_seed)
        torch.manual_seed(self._current_seed)
        torch.cuda.manual_seed_all(self._current_seed)

    def encode(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        return input_ids.to(self.device), len(input_ids[0])

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def set_generation_config(self, seed=-1, temperature=0.7, top_p=0.95, top_k=40, repetition_penalty=1.1, max_length=512, do_sample=True, use_cache=True, ban_eos=False):
        self.generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_length,
            do_sample=do_sample,
            use_cache=use_cache,
            suppress_tokens=ban_eos and [self.tokenizer.eos_token_id] or None
        )
        self.seed = seed

    def pipeline(self, prompts, batch_size=1):
        if not self.pipe:
            # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
            logging.set_verbosity(logging.CRITICAL)
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                generation_config=self.generation_config,
                device=self.device
            )
        self.update_seed()
        answers = []
        with self.do_timing(True) as timing:
            with torch.no_grad():
                # TODO: batch_size >1 causes gibberish output, investigate
                output = self.pipe(prompts, return_tensors=True, batch_size=batch_size)

            for index, gen in enumerate(output):
                tokens = gen[0]['generated_token_ids']
                input_ids, len_input_ids = self.encode(prompts[index])
                len_reply = len(tokens) + 1 - len_input_ids
                response = self.decode(tokens)
                reply_tokens = tokens[-len_reply:]
                reply = self.tokenizer.decode(reply_tokens)

                result = {
                    'response': response,   # The response in full, including prompt
                    'reply': reply,         # Just the reply, no prompt
                    'len_reply': len_reply, # The length of the reply tokens
                    'seed': self.seed,      # The seed used to generate this response
                    'time': timing['time']  # The time in seconds to generate the response
                }
                answers.append(result)

        return answers

    def generate(self, prompt):
        self.update_seed()
        input_ids, len_input_ids = self.encode(prompt)

        with self.do_timing(True) as timing:
            with torch.no_grad():
                tokens = self.model.generate(inputs=input_ids, generation_config=self.generation_config)[0].cuda()
            len_reply = len(tokens) - len_input_ids
            response = self.tokenizer.decode(tokens)
            reply_tokens = tokens[-len_reply:]
            reply = self.tokenizer.decode(reply_tokens)
        
        result = {
            'response': response,   # The response in full, including prompt
            'reply': reply,         # Just the reply, no prompt
            'len_reply': len_reply, # The length of the reply tokens
            'seed': self.seed,      # The seed used to generate this response
            'time': timing['time']  # The time in seconds to generate the response
        }
        return result

class HuggingFaceModel(LanguageModel):
    def create_model(self, model_dir, load_in_8bit):
        with self.do_timing(self.timing, "load HF model without accelerate and 8bit = {args.HF_8bit}") as timing:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map='auto', load_in_8bit=load_in_8bit)

class HuggingFaceAcceleratedModel(LanguageModel):
    def create_model(self, model_dir, **kwargs):
        with self.do_timing(self.timing, "load HF model with accelerate") as timing:
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, **kwargs)

            model.tie_weights()

            self.model = load_checkpoint_and_dispatch(model, model_dir, device_map="auto")

class GPTQModel(LanguageModel):
    def create_model(self, model_dir, model_basename=None, bits=4, group_size=128, desc_act=False,
                     fused_attn=True, use_triton=False, warmup_triton=True, fused_mlp=False, use_cuda_fp16=True, trust_remote_code=False):
        if not isfile(f"{model_dir}/quantize_config.json"):
            quantize_config = BaseQuantizeConfig(
                    bits=bits, group_size=group_size, desc_act=desc_act
                )
        else:
            quantize_config = None

        #use_safetensors = isfile(f"{model_dir}/{model_basename}.safetensors")
        use_safetensors=True

        self.timing = True
        with self.do_timing(self.timing, "load GPTQ model") as timing:
            self.model = AutoGPTQForCausalLM.from_quantized(model_dir,
                    use_safetensors=use_safetensors,
                    #max_memory={ 0: "4GiB", "cpu": "50GiB" },
                    #full_cpu_offload=True,
                    model_basename=model_basename,
                    device=self.device,
                    inject_fused_attention=fused_attn,
                    inject_fused_mlp=fused_mlp,
                    trust_remote_code=trust_remote_code,
                    use_triton=use_triton,
                    use_cuda_fp16=use_cuda_fp16,
                    warmup_triton=warmup_triton and use_triton,
                    quantize_config=quantize_config)


def benchmark(llm, args, prompts):
    num_inf = args.bench
    bench = {'time_per_token': [], 'tokens_per_second': [], 'seed': [],
             'prompt_length': [], 'response_length': []}
    progress_bar = tqdm(prompts)
    progress_bar.set_description(f"Inference @ --.-- tok/s")

    for prompt in progress_bar:
        result = llm.generate(prompt)
        #result = llm.pipeline(prompt)
        len_reply = result['len_reply']
        output = result['reply']
        seed = result['seed']
        time = result['time']
        if not args.quiet:
            tqdm.write(f"Prompt:\n{prompt}")

        time_per_token = time / len_reply
        tokens_per_sec = len_reply / time

        bench['time_per_token'].append(time_per_token)
        bench['tokens_per_second'].append(tokens_per_sec)
        bench['average_tok_s'] = sum(bench['tokens_per_second']) / len(bench['time_per_token'])
        progress_bar.set_description(f"Inference @ {bench['average_tok_s']:.2f} tok/s")

        if args.csv:
            bench['seed'].append(seed)
            bench['prompt_length'].append(len(prompt))
            bench['response_length'].append(len_reply)

        if not args.quiet:
            tqdm.write(output + "\n")

    if args.timing:
        print("Inference timing:")
        avg_time_per_token = f"{sum(bench['time_per_token']) / len(bench['time_per_token']):.3f}"
        print(f"Average over {len(prompts)} runs: {bench['average_tok_s']:.3f} tokens/s ({avg_time_per_token} secs per token)")

        combined_results = [f'{x:.3f} ({y:.3f}s)' for x, y in zip(bench['tokens_per_second'], bench['time_per_token']) ]
        print(f"Result for {len(prompts)} runs: tok/s (s/tok):", ', '.join(combined_results))

    if args.csv:
        create_csv(bench, args)

def create_csv(bench, args):
    # Define the column headers
    headers = ['Model type', 'Model Dir', 'Model Basename', 'bits', 'group_size', 'desc_act',
               'Triton', 'Fused Attention', 'Fused MLP', 'no_cuda_fp16', 'seed',
               'Prompt Length (chars)', 'Response Length (chars)', 'Tokens/s', 'Seconds/token']

    # Create the rows for the CSV file
    rows = []
    for seed, tok_s, time_tok, p_len, r_len in zip(bench['seed'],
                                                bench['tokens_per_second'],
                                                bench['time_per_token'],
                                                bench['prompt_length'],
                                                bench['response_length']):
        row = [args.model_type, args.model_dir, args.model_basename, args.bits, args.group_size,
               args.desc_act, args.use_triton, args.fused_attn, args.fused_mlp, args.no_cuda_fp16,
               seed, p_len, r_len, tok_s, time_tok]
        rows.append(row)

    if args.csv_file:
        csv_filename = args.csv_file

        # Write the results to the CSV file
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            csv_writer.writerows(rows)

        print(f"CSV saved to {csv_filename}")

    else:
        # Print the CSV to the screen
        print("CSV data:")
        print(', '.join(headers))
        for row in rows:
            print(', '.join(str(value) for value in row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference client for AutoGPTQ and HF format models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('model_dir', type=str, help='Local model folder')

    # Prompt arguments
    prompt = parser.add_argument_group('prompt arguments')
    prompt.add_argument('--prompt', '-p', type=str, help='Single prompt to use for inference')
    prompt.add_argument('--file', '-f', type=str, help='File of prompts to load. Each prompt will be run in turn.')
    prompt.add_argument('--prompt_template', '-pt', type=str, default=list(prompt_templates.keys())[0], choices=list(prompt_templates.keys()), help=f'Prompt template to use.')

    # Model arguments
    model = parser.add_argument_group('model arguments')
    model.add_argument('--model_type', '-mt', type=str, default="GPTQ", choices=['GPTQ', 'HF', 'HF-accel'], help='Type of model to load.')
    model.add_argument('--device', '-d', type=str, default="cuda:0", help='Device to load model on')
    model.add_argument('--tokenizer_dir', '-td', type=str, help='Specify tokenizer directory')
    model.add_argument('--use_fast', action='store_true', help='Use fast tokenizer')
    model.add_argument('--HF_8bit', action='store_true', help='Use bitsandbytes 8bit inference for HF models. Only for --model_type HF')

    # Inference arguments
    inference = parser.add_argument_group('inference arguments')
    inference.add_argument('--temperature', '--temp', type=float, default=0.7, help='Inference temperature')
    inference.add_argument('--top_k', type=int, default=40, help='Inference top_k')
    inference.add_argument('--top_p', type=float, default=0.95, help='Inference top_p')
    inference.add_argument('--repetition_penalty', type=float, default=1.1, help='Inference repetition_penalty')
    inference.add_argument('--max_length', '-ml', type=int, default=512, help='Max length of returned tokens.')
    inference.add_argument('--seed', type=int, default=-1, help='Seed to use for generation. Default of -1 generates a random seed each time.')
    inference.add_argument('--ban_eos', action="store_true", help='Ban the End-of-String token. Ensures results will always be at max length.')

    # GPTQ arguments
    gptq = parser.add_argument_group('GPTQ arguments')
    gptq.add_argument('--model_basename', '-mb', type=str, help='Model basename, ie the full name of the model file before .safetensors or .bin')
    gptq.add_argument('--bits', '-b', type=int, default=4, help='Quantize bits. Specify if model has no quantize_config.json')
    gptq.add_argument('--group_size', '-gs', type=int, default=128, help='Quantize group_size. Specify if model has no quantize_config.json')
    gptq.add_argument('--desc_act', '-da', action='store_true', help='Quantized model uses desc_act/--act-order. Specify for act-order models if model has no quantize_config.json')
    gptq.add_argument('--use_triton', action='store_true', help='Uses Triton if set, otherwise uses CUDA')
    gptq.add_argument('--triton_no_warmup', action='store_true', help='Triton: do not warm up autotune cache')
    gptq.add_argument('--fused_attn', '-fa', action='store_true', help='Enable quant_attn to speed up Triton and CUDA inference')
    gptq.add_argument('--fused_mlp', '-fm', action='store_true', help='Enable fused_mlpto speed up Triton inference')
    gptq.add_argument('--no_cuda_fp16', action='store_true', help='Do not CUDA fp16 support. May improve perf on older NVidia architectures?')
    gptq.add_argument('--trust_remote_code', action='store_true', help='Trust remote code')

    # Timing and benchmarking arguments
    timing = parser.add_argument_group('timing and timinging arguments')
    timing.add_argument('--timing', '-t', action='store_true', help='Show timing')
    timing.add_argument('--bench', action="store_true",  help='Benchmark inference by repeatedly running prompts')
    timing.add_argument('--bench_mode', type=str, default='loop', choices=['loop', 'repeat'], help='Benchmark: how to repeat prompts. Given bench_count=2 and prompts ABCD, "loop" = ABCDABCD, "repeat" = AABBCCDD')
    timing.add_argument('--bench_count', type=int, default=10, help='Benchmark inference by running prompts the specified number of times')
    timing.add_argument('--csv', action="store_true", help='Generate a CSV of timing results')
    timing.add_argument('--csv_file', type=str, help='Save the CSV results to the specified filename instead of printing to screen')
    timing.add_argument('--quiet', '-q', action='store_true', help='Do not show the result of running the prompt.')

    args = parser.parse_args()

    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    if args.HF_8bit and not args.model_type == "HF":
        raise ValueError("--8bit can only be used with --model_type 'HF'. It is not supported for GPTQ, or HF-accel")

    if not args.prompt and not args.file:
        raise ValueError("Must specify one of --prompt/-p or --file/-f")

    prompts = []

    if args.prompt:
        prompts.append(args.prompt)

    if args.file:
        if not isfile(args.file):
            raise FileNotFoundError(f"Could not load prompt file {args.file}")

        with open(args.file, 'r') as prompt_file:
            prompts += [line.rstrip() for line in prompt_file]

    if args.model_type.upper() == "GPTQ":
        llm = GPTQModel(args.model_dir, args.device, tokenizer_dir=args.tokenizer_dir, use_fast_tokenizer=args.use_fast, 
                        model_basename = args.model_basename,
                        bits = args.bits,
                        group_size = args.group_size,
                        desc_act = args.desc_act,
                        fused_attn = args.fused_attn,
                        use_triton = args.use_triton,
                        warmup_triton  = not args.triton_no_warmup,
                        fused_mlp = args.fused_mlp,
                        use_cuda_fp16 = not args.no_cuda_fp16,
                        trust_remote_code = args.trust_remote_code)
    elif args.model_type.upper() == "HF-ACCEL":
        llm = HuggingFaceAcceleratedModel(args.model_dir, args.device, tokenizer_dir=args.tokenizer_dir, use_fast_tokenizer=args.use_fast, trust_remote_code=args.trust_remote_code)
    else:
        llm = HuggingFaceModel(args.model_dir, args.device, tokenizer_dir=args.tokenizer_dir, use_fast_tokenizer=args.use_fast, load_in_8bit=args.HF_8bit, trust_remote_code=args.trust_remote_code)

    # Pass inference arguments
    inference_args = [arg.dest for arg in inference._group_actions]
    filtered_args = {k: v for k, v in vars(args).items() if k in inference_args}
    llm.set_generation_config(**filtered_args)

    llm.timing = True # args.timing

    prompts = [get_prompt(args.prompt_template, prompt) for prompt in prompts]
    if args.bench:
        if args.bench_mode == "loop":
            prompts = prompts * args.bench_count
        elif args.bench_mode == "repeat":
            prompts = [prompt for prompt in prompts for _ in range(args.bench_count)]

        benchmark(llm, args, prompts)
    elif len(prompts) > 1:
        answers = llm.pipeline(prompts)
        pprint(answers)
        #print(output)
    else:
        for prompt in prompts:
            prompt = get_prompt(args.prompt_template, prompt)
            output = llm.pipeline(prompt)
            #output = llm.generate(prompt)
            print(output['response'])
