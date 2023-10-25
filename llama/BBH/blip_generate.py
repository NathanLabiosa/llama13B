import json
import os
import re
import argparse
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch
import csv

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the test folder')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder')
args = parser.parse_args()

# Load the InstructBLIP model and processor
model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path)
processor = InstructBlipProcessor.from_pretrained(args.model_path)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)  # replace 'model_name_or_path' with the name or path of your model


# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def load_chain_of_thought_prompts(txt_file_path):
    with open(txt_file_path, 'r') as f:
        return f.read()

bbh_path = os.path.join(args.data_path, 'bbh')
cot_prompts_path = os.path.join(args.data_path, 'cot-prompts')

# Use a blank image for every input
blank_image = Image.new('RGB', (224, 224), color='black')

for filename in os.listdir(bbh_path):
    if filename.endswith('.json'):
        input_file_path = os.path.join(bbh_path, filename)
        txt_file_path = os.path.join(cot_prompts_path, filename.replace('.json', '.txt'))
        few_shot_prompt = load_chain_of_thought_prompts(txt_file_path)
        # few_shot_prompt = ""
        output_dir = os.path.join(args.output_path, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, filename.replace('.json', '_output.jsonl'))

        with open(input_file_path, 'r') as jsonfile:
            data = json.load(jsonfile)
            examples = data['examples']

            for example_data in examples:
                stem = example_data['input']
                answerKey = example_data['target']

                match = re.search(r"Options:\n(.+?)$", example_data['input'], re.DOTALL)
                options_list = [option.strip() for option in match.group(1).split("\n") if option.strip()] if match else []

                prompt = f"'{few_shot_prompt}' '{stem}'"

                truncated_tokens = tokenizer.encode(prompt, max_length=512, truncation=True)
                if len(truncated_tokens) > 500:
                    truncated_tokens = truncated_tokens[-500:]
                truncated_input = tokenizer.decode(truncated_tokens)
                inputs = processor(images=blank_image, text=truncated_input, return_tensors="pt").to(device)

                # Generate a response
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=256,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                ) 
                rationale = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                #rationale = rationale.replace(prompt, '')

                output_example = {
                    'question': stem,
                    'answerKey': answerKey,
                    'model_answer': rationale
                }
                if options_list:
                    output_example['options'] = options_list

                with open(output_file_path, 'a') as writer:
                    writer.write(json.dumps(output_example) + '\n')
