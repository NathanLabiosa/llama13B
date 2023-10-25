import json
import jsonlines
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
import torch

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
        'question': 'Where can you buy binoculars?',
        'choices': [{'label': 'A', 'text': 'sporting goods store'}, {'label': 'B', 'text': 'suitcase'}, {'label': 'C', 'text': 'backpack'}],
        'label': 'A'
    },
    {
        'question': 'One of the potential hazards of attending school is what?',
        'choices': [{'label': 'A', 'text': 'taking tests'}, {'label': 'B', 'text': 'get smart'}, {'label': 'C', 'text': 'colds and flu'}],
        'label': 'C'
    },
    {
        'question': 'If somebody buys something and gives it to me as a free gift, what is the cost status of the gift?',
        'choices': [{'label': 'A', 'text': 'imprisoned'}, {'label': 'B', 'text': 'paid for'}, {'label': 'C', 'text': 'expensive'}],
        'label': 'B'
    }
]

few_shot_prompt = "\n\n".join([
    f"Given the question: '{example['question']}', the choices: {', '.join(['(' + choice['label'] + ') ' + choice['text'] for choice in example['choices']])}, the correct answer is ({example['label']}) {next(choice['text'] for choice in example['choices'] if choice['label'] == example['label'])}."
    for example in examples
])



# Open the JSONL file
with jsonlines.open('/home/nlabiosa/llama13B/llama/cos-e_experimentation/data.jsonl', mode='r') as reader:
    # Iterate over each example in the dataset
    for i, example in enumerate(reader):
        stem = example['question']['stem']
        answerKey = example['answerKey']

        # Convert the answerKey to answer text
        choices = example['question']['choices']
        for choice in choices:
            if choice['label'] == answerKey:
                answer = choice['text']

        # Format the choices text
        choices_text = ', '.join([f'({choice["label"]}) {choice["text"]}' for choice in choices])

        # Combine the stem and answer in a question-answer format
        prompt = f"{few_shot_prompt} Given the question: '{stem}', the choices: {choices_text}, the correct answer is"

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

        # Add the rationale to the example
        example['model_answer'] = rationale

        with open('blip.jsonl', 'a') as writer:
            writer.write(json.dumps(example) + '\n')

