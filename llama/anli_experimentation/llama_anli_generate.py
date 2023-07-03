import json
import jsonlines
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
from tqdm import tqdm
import shortuuid

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math


# Load the LLMa model and tokenizer
model_name = "/home/nlabiosa/llama13B/llama/llama-13b-hf" # replace with the path to your model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()


# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device, flush = True)
# Move the model to the device
model = model.to(device)


# Label mapping
label_mapping = {
    'e': "entailment",
    'n': "neutral",
    'c': "contradiction"
}

# Hardcoded examples for few-shot learning
examples = [
    {
        'context': 'Linguistics is the scientific study of language, and involves an analysis of language form, language meaning, and language in context. The earliest activities in the documentation and description of language have been attributed to the 4th century BCE Indian grammarian Pāṇini, who wrote a formal description of the Sanskrit language in his "Aṣṭādhyāyī".',
        'hypothesis': 'Form and meaning are the only aspects of language linguistics is concerned with.',
        'label': 'c',
        'rationale': 'Linguistics involves an analysis of language form, language meaning, and language in context, so context is also a crucial aspect.'
    },
    {
        'context': 'Franco Zeffirelli, KBE Grande Ufficiale OMRI (] ; born 12 February 1923) is an Italian director and producer of operas, films and television. He is also a former senator (1994–2001) for the Italian centre-right "Forza Italia" party. Recently, Italian researchers have found that he is one of the few distant relatives of Leonardo da Vinci.',
        'hypothesis': 'Franco Zeffirelli had a political career',
        'label': 'e',
        'rationale': 'Franco Zeffirelli was a senator so he had a political career.'
    },
    {
        'context': 'Almost Sunrise is a 2016 American documentary film directed by Michael Collins. It recounts the story of two Iraq veterans, Tom Voss and Anthony Anderson, who, in an attempt to put their combat experience behind them, embark on a 2,700-mile trek on foot across America. It made its world premiere on the opening night of the Telluride Mountainfilm Festival on 27 May, 2016.',
        'hypothesis': 'Tom and Anthony have both killed someone.',
        'label': 'n',
        'rationale': "The prompt references combat experience, but is vague, so you don't know if that entailed killing anyone."
    }

]

# In the few-shot examples
few_shot_prompt = "\n\n".join([
    f"Given the context: '{example['context']}', the hypothesis: '{example['hypothesis']}', it is a '{label_mapping[example['label']]}'. "
    for example in examples
])




# Open the JSONL file
with jsonlines.open('/home/nlabiosa/llama13B/llama/anli/data/anli_v1.0/R1/test.jsonl', mode='r') as reader, open('anli_rationales.jsonl', 'a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(reader):
        context = example['context']
        hypothesis = example['hypothesis']
        answer = label_mapping[example['label']]  # Convert the label to its corresponding relation


        # Combine the context, hypothesis, and answer in a question-answer format
        prompt = f"{few_shot_prompt}\n\nGiven the context: '{context}', the hypothesis: '{hypothesis}', it is a  "


        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        # Move the inputs to the device
        inputs = inputs.to(device)

        # Generate a response
        #with torch.inference_mode():
        outputs = model.generate(inputs, max_length=600)

        # Decode the output tokens to text
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the few-shot examples from the output
        output_text = output_text.replace(few_shot_prompt, '')

        # Split the output into the predicted relation and the rationale
        #predicted_relation, rationale = output_text.split(" because ")

        # Add the predicted relation and the rationale to the example
        #example['predicted_relation'] = predicted_relation
        example['rationale'] = output_text

        with open('llama_anli_generate_fulltest.jsonl', 'a') as writer:
            writer.write(json.dumps(example) + '\n')
        # Print a status update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples")

print("Processing complete!")
