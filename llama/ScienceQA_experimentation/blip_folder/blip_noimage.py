import json
import jsonlines
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
import torch
import os

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


examples = [
    {
        'question': "What does the simile in this text suggest?\nTara rubbed coconut oil on her hands, which were like the parched earth during a drought.\nContext: N/A\nOptions: (A) Tara was baking something. (B) Tara's hands were dry and cracked.",
        'answer': "B",
        'lecture': 'Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\nA simile uses like or as to compare two things that are not actually alike.\nThe cat\'s fur was as dark as the night.\nSOLUTION: The text includes a simile, using like or as to compare two things that are not actually alike.\nThe simile like the parched earth during a drought suggests that Tara\'s hands were dry and cracked. A drought is a period without rain; the ground during a drought can become hard and cracked.'
    }
]



# Open the JSON file
with open('/home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/llava_minitest_QCM-LEPA.json', 'r') as f:
    data = json.load(f)

# Open the output file
with open('blip_noimage.jsonl', 'a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(data):
        question = example['conversations'][0]
        qs = question['value']
        qs = qs.replace('<image>', '').strip()

        few_shot_prompt = ""

        for ex in examples:
            few_shot_prompt += f"{ex['question']} Think step by step and justify your steps. {ex['lecture']} The answer is {ex['answer']} \n"

        # Concatenate the question and answer in a question-answer format
        prompt = few_shot_prompt + qs + " . Think step by step and justify your steps."

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

        # Add the generated response to the example
        example['model_answer'] = rationale

        with open('blip_noimage.jsonl', 'a') as writer:
            writer.write(json.dumps(example) + '\n')

