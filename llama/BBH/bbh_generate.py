import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re

import csv
from llava.conversation import default_conversation
from llava.utils import disable_torch_init


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def load_chain_of_thought_prompts(txt_file_path):
    with open(txt_file_path, 'r') as f:
        # Skip the first two lines (canary and separator)
        f.readline()
        f.readline()
        # Read the rest of the file
        return f.read()




@torch.inference_mode()
def eval_model(model_name, data_path, output_path, args):

    # Define the paths to the 'bbh' and 'cot-prompts' folders
    bbh_path = os.path.join(args.data_path, 'bbh')
    cot_prompts_path = os.path.join(args.data_path, 'cot-prompts')
    
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16).cuda()


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
                    qs = prompt

                    conv = default_conversation.copy()
                    conv.append_message(conv.roles[0], qs)
                    prompt = conv.get_prompt()
                    inputs = tokenizer([prompt])
                    input_ids = torch.as_tensor(inputs.input_ids).cuda()
                    stopping_criteria = KeywordsStoppingCriteria([conv.sep], tokenizer, input_ids)
                    output_ids = model.generate(
                        input_ids,
                        do_sample=True,
                        use_cache=True,
                        temperature=0.7,
                        max_new_tokens=512,
                        stopping_criteria=[stopping_criteria])
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    try:
                        index = outputs.index(conv.sep, len(prompt))
                    except ValueError:
                        outputs += conv.sep
                        index = outputs.index(conv.sep, len(prompt))

                    outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()
                    outputs = outputs.replace(prompt, "")
                    output_example = {
                        'question': stem,
                        'answerKey': answerKey,
                        'model_answer': outputs
                    }
                    
                    # Add options to the output_example if they exist
                    if options_list:
                        output_example['options'] = options_list

                    # Write the example to the JSONL file
                    with open(output_file_path, 'a') as writer:
                        writer.write(json.dumps(output_example) + '\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the test folder')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder')
    args = parser.parse_args()
    

    eval_model(args.model_name, args.data_path, args.output_path, args)
