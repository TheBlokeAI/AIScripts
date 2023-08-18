#
# A simple AutoGPTQ / GPTQ test script
# Note this has not been tested recently - last used with AutoGPTQ 0.2.2
#

from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging
import argparse

parser = argparse.ArgumentParser(description='AutoGPTQ testing script')
parser.add_argument('model_name_or_path', type=str, help='Model folder or repo')
parser.add_argument('--model_basename', type=str, help='Model file basename if model is not named gptq_model-Xb-Ygr')
parser.add_argument('--use_slow', action="store_true", help='Use slow tokenizer')
parser.add_argument('--use_safetensors', action="store_true", help='Model file basename if model is not named gptq_model-Xb-Ygr')
parser.add_argument('--use_triton', action="store_true", help='Use Triton for inference?')
parser.add_argument('--trust_remote_code', action="store_true", help='Trust remote code. Required for some new model types without native transformers support.')
parser.add_argument('--bits', type=int, default=4, help='Specify GPTQ bits. Only needed if no quantize_config.json is provided')
parser.add_argument('--group_size', type=int, default=128, help='Specify GPTQ group_size. Only needed if no quantize_config.json is provided')
parser.add_argument('--desc_act', action="store_true", help='Specify GPTQ desc_act. Only needed if no quantize_config.json is provided')
parser.add_argument('--device', type=str, default="cuda:0", help='Device to load model on to')
parser.add_argument('--min_new_tokens', type=int, default=512, help='Device to load model on to')
parser.add_argument('--max_new_tokens', type=int, default=512, help='Device to load model on to')

args = parser.parse_args()

logger = logging.getLogger()

logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
        )

quantized_model_dir = args.model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=not args.use_slow)

try:
   quantize_config = BaseQuantizeConfig.from_pretrained(quantized_model_dir)
except FileNotFoundError:
    quantize_config = BaseQuantizeConfig(
            bits=args.bits,
            group_size=args.group_size,
            desc_act=args.desc_act
        )

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
        use_safetensors=True,
        model_basename=args.model_basename,
        inject_fused_attention=False,
        inject_fused_mlp=False,
        device=args.device,
        use_triton=args.use_triton,
        quantize_config=quantize_config)

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

prompt = "Tell me about AI"
prompt_template=f'''### Human: {prompt}
### Assistant:'''

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    min_new_tokens=args.min_new_tokens,
    max_new_tokens=args.max_new_tokens,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=args.max_new_tokens, min_new_tokens=args.min_new_tokens)
print(tokenizer.decode(output[0]))

