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
df = pd.read_csv("/home/nlabiosa/llama13B/llama/SVAMP_experimentation/data.csv")

# List of hardcoded examples
examples = [

    {
        'question': '7 red apples and 2 green apples are in the basket . how many apples are in the basket ?',
        'answer': '9 apples .'
    },

    {
        'question': 'jake has 6 fewer peaches than steven . steven has 13 peaches . how many peaches does jake have ?',
        'answer': '7 peaches .'
    },

    {
        'question': 'kim has 4 cousins . she wants to give each one 5 pieces of gum . how much gum will she need ?',
        'answer': '20 pieces .'
    },
    {
        'question': 'mrs. hilt has 50 cents . a pencil costs 5 cents . how many pencils can she buy with the money she has ?',
        'answer': '10 pencils .'
    }
]

# Open the output file
with open("blip_svamp.jsonl", 'w') as file:

    # Iterate over each example in the dataset
    for i, example in df.iterrows():
        # Split the numbers and create a mapping from 'numberN' to the corresponding number
        numbers = example['Numbers'].split()
        number_mapping = {f"number{n}": numbers[n] for n in range(len(numbers))}

        # Replace 'numberN' in the Question with the corresponding numbers
        question = example['Question']
        for number, value in number_mapping.items():
            question = question.replace(number, value)

        # Get the answer from the dataframe
        answer = example['Answer']

        few_shot_prompt = ""
        for ex in examples:
            ex_question = ex['question']
            ex_answer = ex['answer']

            prompt = f"\nGiven the question: '{ex_question}', the answer is: '{ex_answer}'."
            few_shot_prompt += prompt


        # Combine the question and answer into a prompt for the model
        prompt = few_shot_prompt + f"Given the question: '{question}', and the answer is: "

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
        with open('blip_svamp.jsonl', 'a') as writer:
            writer.write(json.dumps(example_dict) + '\n')

