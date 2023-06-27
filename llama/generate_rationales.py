import json
import jsonlines
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


# Load the LLMa model and tokenizer
model_name = "llama-13b-hf" # replace with the path to your model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Label mapping
label_mapping = {
    'e': "entailment",
    'n': "neutral",
    'c': "contradiction"
}


# Open the JSONL file
with jsonlines.open('anli/data/anli_v1.0/R1/train.jsonl', mode='r') as reader, open('rationales.jsonl', 'a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(reader):
        context = example['context']
        hypothesis = example['hypothesis']
        answer = label_mapping[example['label']]  # Convert the label to its corresponding relation


        # Combine the context, hypothesis, and answer in a question-answer format
        prompt = f"Given the context: '{context}', the hypothesis: '{hypothesis}', why is the answer: '{answer}'"
        #prompt = f"Given the context: '{context}', is the hypothesis: '{hypothesis}' correct? Why or why not?"

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Generate a response
        outputs = model.generate(inputs, max_length=400)

        # Decode the output tokens to text
        rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Add the rationale to the example
        example['rationale'] = rationale

        with open('rationales.jsonl', 'a') as writer:
            writer.write(json.dumps(example) + '\n')
        # Print a status update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples")

print("Processing complete!")
