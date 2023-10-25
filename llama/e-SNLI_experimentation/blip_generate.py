import json
import jsonlines
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
import torch
import pandas as pd

# Load the InstructBLIP model and processor
model = InstructBlipForConditionalGeneration.from_pretrained("/home/nlabiosa/llama13B/llama/blip")
processor = InstructBlipProcessor.from_pretrained("/home/nlabiosa/llama13B/llama/blip")

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Use the blank image for every input
# Create a blank white image
width, height = 224, 224  # or any other desired size
blank_image = Image.new('RGB', (width, height), color='white')
raw_image = blank_image

# Read the CSV file
df = pd.read_csv("/home/nlabiosa/llama13B/llama/e-SNLI_experimentation/esnli_minitest.csv")

# List of hardcoded examples
examples = [
    {
        'gold_label': 'entailment',
        'Sentence1': 'Two young children in blue jerseys, one with the number 9 and one with the number 2 are standing on wooden steps in a bathroom and washing their hands in a sink.',
        'Sentence2': 'Two kids in numbered jerseys wash their hands.',
        'Explanation_1': 'Young children are kids. Jerseys with number 9 and 2 are numbered jerseys.'
    },
    {
        'gold_label': 'contradiction',
        'Sentence1': 'A young boy in a field of flowers carrying a ball',
        'Sentence2': 'dog in pool',
        'Explanation_1': 'The boy cannot also be a dog.'
    },
    {
        'gold_label': 'neutral',
        'Sentence1': 'A man in a green jersey and rollerskates stumbles as a man in a black jersey appears to collide with him.',
        'Sentence2': 'They both fall to the ground.',
        'Explanation_1': 'The combination of stumbling and colliding may not leave them both on the ground.',
    }
]

# Open the output file
with open("blip_esnli.jsonl", 'w') as file:

    # Iterate over each example in the dataset
    for i, example in df.iterrows():
        Sentence1 = example['Sentence1']
        Sentence2 = example['Sentence2']
        label = example['gold_label']  # Convert the label to its corresponding relation

        # Combine the premise, hypothesis, and label in a question-answer format
        prompt = f"Given the two sentences: '{Sentence1}' '{Sentence2}', their relationship is "
        few_shot_prompt = ""

        # Prepend the prompt with a few examples
        for ex in examples:
            ex_Sentence1 = ex['Sentence1']
            ex_Sentence2 = ex['Sentence2']
            ex_label = ex['gold_label']

            few_shot_prompt = few_shot_prompt + (f"Given the two sentences: '{ex_Sentence1}' '{ex_Sentence2}', "
                        f"their relationship is '{ex_label}'. ")
        prompt = few_shot_prompt + prompt

        # Process the blank image and the prompt
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)

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

        example_dict = example.to_dict()
        example_dict['model_answer'] = rationale

        # Write the example (with the added rationale) to the JSON Lines file
        with open('blip_esnli.jsonl', 'a') as writer:
            writer.write(json.dumps(example_dict) + '\n')

