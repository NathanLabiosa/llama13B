import json
import jsonlines
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the LLMa model and tokenizer
model_name = "llama-13b-hf" # replace with the path to your model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Open the JSONL file
with jsonlines.open('/home/nlabiosa/llama13B/llama/cos-e/data/v1.0/train_rand_split.jsonl', mode='r') as reader, jsonlines.open('cos-e_rationales.jsonl', mode='a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(reader):
        stem = example['question']['stem']
        answerKey = example['answerKey']

        # Convert the answerKey to answer text
        choices = example['question']['choices']
        for choice in choices:
            if choice['label'] == answerKey:
                answer = choice['text']

        # Combine the stem and answer in a question-answer format
        prompt = f"Given the statement: '{stem}', why is the answer: '{answer}'?"

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Generate a response
        outputs = model.generate(inputs, max_length=100)

        # Decode the output tokens to text
        rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Add the rationale to the example
        example['rationale'] = rationale

        # Write the example back to the JSONL file
        with open('cos-e_rationales.jsonl', 'a') as writer:
            writer.write(json.dumps(example) + '\n')

        # Print a status update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples")

print("Processing complete!")
