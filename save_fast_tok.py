#
# Save a fast tokenizer for a Llama model that only has tokenizer.model
# requires protobf==3.20.0
#

import argparse
import os

# Requires protobuf==3.20.0
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='convert slow tokenizer to fast')
parser.add_argument('model_dir', type=str, help='Local model folder')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

tokenizer.save_pretrained(args.model_dir)
