import csv
import torch
import os
import json
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the test folder')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder')
args = parser.parse_args()

# Load the LLMa model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, use_cache=True).cuda()

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

def load_chain_of_thought_prompts(txt_file_path):
    with open(txt_file_path, 'r') as f:
        # Skip the first two lines (canary and separator)
        f.readline()
        f.readline()
        # Read the rest of the file
        return f.read()

@torch.inference_mode()
def run():

    # Define the paths to the 'bbh' and 'cot-prompts' folders
    bbh_path = os.path.join(args.data_path, 'bbh')
    cot_prompts_path = os.path.join(args.data_path, 'cot-prompts')
    
        # Iterate through all JSON files in the 'bbh' folder
    for filename in os.listdir(bbh_path):
        if filename.endswith('.json'):
            input_file_path = os.path.join(bbh_path, filename)
            
            # Load the corresponding chain of thought prompts from the .txt file in 'cot-prompts' folder
            txt_file_path = os.path.join(cot_prompts_path, filename.replace('.json', '.txt'))
            few_shot_prompt = load_chain_of_thought_prompts(txt_file_path)

            # Define the output directory
            output_dir = os.path.join(args.output_path, 'output')
            os.makedirs(output_dir, exist_ok=True)

            # Define the path for the output JSONL file
            output_file_path = os.path.join(output_dir, filename.replace('.json', '_output.jsonl'))

            # Load the JSON content
            with open(input_file_path, 'r') as jsonfile:
                data = json.load(jsonfile)
                examples = data['examples']

                for example_data in examples:
                    stem = example_data['input']
                    answerKey = example_data['target']

                    # Check if "Options:" is present in the stem
                    match = re.search(r"Options:\n((?:\([A-F]\) [^\n]+\n)+)", example_data['input'])
        
                    # Extract the options using regex
                    match = re.search(r"Options:\n(.+?)$", example_data['input'], re.DOTALL)
                    
                    if match:
                        options_str = match.group(1)
                        # Split the options into a list
                        options_list = [option.strip() for option in options_str.split("\n") if option.strip()]
                    else:
                        options_list = []

                    # Combine the stem and answer in a question-answer format
#                    prompt = f"'{few_shot_prompt}' '{stem}'"
                    prompt = stem

                    # Tokenize the prompt
                    inputs = tokenizer.encode(prompt, return_tensors='pt')
                    inputs = inputs.to(device)

                    # Generate a response
                    outputs = model.generate(inputs, max_new_tokens=256)

                    # Decode the output tokens to text
                    rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Remove the prompt from the rationale
                    rationale = rationale.replace(prompt, "")


                    output_example = {
                        'question': stem,
                        'answerKey': answerKey,
                        'model_answer': rationale
                    }
                    
                    # Add options to the output_example if they exist
                    if options_list:
                        output_example['options'] = options_list

                    # Write the example to the JSONL file
                    with open(output_file_path, 'a') as writer:
                        writer.write(json.dumps(output_example) + '\n')
    

run()
