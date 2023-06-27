import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


# Load the LLMa model and tokenizer
model_name = "llama-13b-hf" # replace with the path to your model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)


# Load ANLI dataset
with open('anli/data/anli_v1.0/R1/test.jsonl', 'r') as f:
    anli_dataset = [json.loads(line) for line in f]

rationales = []
# Iterate over the dataset
for i, entry in enumerate(anli_dataset):
    hypothesis = entry['hypothesis']
    context = entry['context']

    # Prepare the prompt for the model
    prompt = context + " " + hypothesis

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate a rationale
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_ids, max_length=200)  # you may need to adjust max_length based on your use case

    # Decode the generated ids to get the rationale
    rationale = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Add the rationale to the entry and append it to rationales list
    entry["rationale"] = rationale
    rationales.append(entry)

# Save rationales to a new jsonl file
with open('rationales.jsonl', 'w') as f:
    for rationale in rationales:
        f.write(json.dumps(rationale) + '\n')
