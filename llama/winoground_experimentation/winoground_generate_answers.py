import json
import jsonlines
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the jsonl data')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
args = parser.parse_args()

# Load the LLMa model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, use_cache=True).cuda()


# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)


# Open the JSONL file
with jsonlines.open(args.data_path, mode='r') as reader, jsonlines.open(args.output_path, mode='a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(reader):
        idx = example["id"]
        caption_0 = example["caption_0"]
        caption_1 = example["caption_1"]
        few_shot_prompt = "Is the correct caption (A) or (B)? The correct caption is (A) . Is the correct caption (A) or (B)? The correct caption is (B) . "
        prompt = f"Is the correct caption (A) or (B)? (A) '{caption_0}' (B) '{caption_1}'. The correct caption is "
        prompt = few_shot_prompt + prompt
        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(device)

        # Generate a response
        outputs = model.generate(inputs, max_new_tokens=20)

        # Decode the output tokens to text
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        output = output.replace(prompt, 'The correct caption is')

        # Add the rationale to the example
        example['text'] = output

        # Write the example back to the JSONL file
        with open(args.output_path, 'a') as writer:
            writer.write(json.dumps(example) + '\n')

        # Print a status update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples")

print("Processing complete!")
