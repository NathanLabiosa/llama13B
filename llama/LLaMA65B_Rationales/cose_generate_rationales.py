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


examples = [
    {
        'question': 'Where can you buy binoculars?',
        'choices': [{'label': 'A', 'text': 'sporting goods store'}, {'label': 'B', 'text': 'suitcase'}, {'label': 'C', 'text': 'backpack'}],
        'label': 'A',
	'explaination': "Binaculars are a good used for viewing distant objects. Goods can be purchased at stores and markets. The correct answer is A sporting goods store."
    },
    {
        'question': 'One of the potential hazards of attending school is what?',
        'choices': [{'label': 'A', 'text': 'taking tests'}, {'label': 'B', 'text': 'get smart'}, {'label': 'C', 'text': 'colds and flu'}],
        'label': 'C',
	'explaination': "Hazards are dangers and risks. Colds and flus are illness that can pose risks to students at school. The correct answer is C colds and flu."
    },
    {
        'question': 'If somebody buys something and gives it to me as a free gift, what is the cost status of the gift?',
        'choices': [{'label': 'A', 'text': 'imprisoned'}, {'label': 'B', 'text': 'paid for'}, {'label': 'C', 'text': 'expensive'}],
        'label': 'B',
	'explaination': 'Someone has already bought the gift, exchanging money for it. Therefore the gift is paid for. The correct answer is B paid for.'
    }
]

few_shot_prompt = "\n\n".join([
    f"'{example['question']}'Choices: {', '.join(['(' + choice['label'] + ') ' + choice['text'] for choice in example['choices']])}, '{example['explaination']}' "
    for example in examples
])


# Open the JSONL file
with jsonlines.open(args.data_path, mode='r') as reader, jsonlines.open(args.output_path, mode='a') as writer:
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
        prompt = f"{few_shot_prompt}'{stem}', Choices: {choices_text},"

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(device)

        # Generate a response
        outputs = model.generate(inputs, max_new_tokens=100)

        # Decode the output tokens to text
        rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the few shot prompt from the rationale
        rationale = rationale[len(few_shot_prompt):]

        # Add the rationale to the example
        example['model_answer'] = rationale

        # Write the example back to the JSONL file
        with open(args.output_path, 'a') as writer:
            writer.write(json.dumps(example) + '\n')

