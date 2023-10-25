import json
import jsonlines
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
import torch
import os
import argparse

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the jsonl data')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
args = parser.parse_args()

# Load the InstructBLIP model and processor
model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path)
processor = InstructBlipProcessor.from_pretrained(args.model_path)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)  # replace 'model_name_or_path' with the name or path of your model

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).to(dtype=torch.bfloat16)

# Use the blank image for every input
# Create a blank white image
width, height = 224, 224  # or any other desired size
blank_image = Image.new('RGB', (width, height), color='black')
raw_image = blank_image



examples = [
    {
        'question': "What does the simile in this text suggest?\nTara rubbed coconut oil on her hands, which were like the parched earth during a drought.\nContext: N/A\nOptions: (A) Tara was baking something. (B) Tara's hands were dry and cracked.",
        'answer': "B",
        'lecture': 'Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\nA simile uses like or as to compare two things that are not actually alike.\nThe cat\'s fur was as dark as the night.\nSOLUTION: The text includes a simile, using like or as to compare two things that are not actually alike.\nThe simile like the parched earth during a drought suggests that Tara\'s hands were dry and cracked. A drought is a period without rain; the ground during a drought can become hard and cracked.'
    }
]



# Open the JSON file
with open(args.data_path, 'r') as f:
    data = json.load(f)

# Open the output file
with open(args.output_path, 'a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(data):
        idx = example["id"]
        question = example['conversations'][0]
        qs = question['value']
        qs = qs.replace('<image>', '').strip()

        # if 'image' in example:
        #     image_file = example["image"]
        #     image_folder = '/home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/test'
        #     raw_image = Image.open(os.path.join(image_folder, image_file))
        # else:
        #     blank_image = Image.new('RGB', (width, height), color='white')
        #     raw_image = blank_image
        blank_image = Image.new('RGB', (width, height), color='black')
        raw_image = blank_image

        # few_shot_prompt = ""

        # for ex in examples:
        #     few_shot_prompt += f"{ex['question']} Think step by step and justify your steps. {ex['lecture']} The answer is {ex['answer']} \n"

        # Concatenate the question and answer in a question-answer format
        # prompt = few_shot_prompt + qs + " . Think step by step and justify your steps."
        prompt = qs + " . Think step by step and justify your steps."

        truncated_tokens = tokenizer.encode(prompt, max_length=512, truncation=True)
        truncated_input = tokenizer.decode(truncated_tokens)
        inputs = processor(images=raw_image, text=truncated_input, return_tensors="pt").to(device).to(dtype=torch.bfloat16)


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
        

        # Decode the output tokens to text
        rationale = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        rationale = rationale.replace(prompt, '')

        # Write the example back to the JSONL file
        with open(args.output_path, 'a') as writer:
            #riter.write(json.dumps(output_text) + '\n')
            writer.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": rationale}) + "\n")


