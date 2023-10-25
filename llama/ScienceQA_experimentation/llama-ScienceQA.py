import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import re


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the jsonl data')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
args = parser.parse_args()

# Load the LLMa model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, use_cache=True).cuda()

choice_tokens = ["A", "B", "C", "D", "E"]
choice_token_ids = tokenizer.convert_tokens_to_ids(choice_tokens)

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

examples = [
    {
        'question': "What information supports the conclusion that Olivia inherited this trait?\nContext: Read the description of a trait.\nOlivia has straight hair.\nOptions: (A) Olivia's neighbor also has straight hair. (B) Olivia's biological parents have red hair. Olivia also has red hair. (C) Olivia's biological mother often wears her straight hair in a ponytail.",
        'answer': "C",
        'lecture': "Organisms, including people, have both inherited and acquired traits. Inherited and acquired traits are gained in different ways.\nInherited traits are passed down from biological parents to their offspring through genes. Genes are pieces of hereditary material that contain the instructions that affect inherited traits. Offspring receive their genes, and therefore gain their inherited traits, from their biological parents. Inherited traits do not need to be learned.\nAcquired traits are gained during a person's life. Some acquired traits, such as riding a bicycle, are gained by learning. Other acquired traits, such as scars, are caused by the environment. Parents do not pass acquired traits down to their offspring."
    },
    {
        'question': "Based on the arrows, which of the following living things is a consumer?\nContext: Below is a food web from an ocean ecosystem. The ecosystem is in Monterey Bay, off the coast of California.\nA food web is a model that shows how the matter eaten by living things moves through an ecosystem. The arrows show how matter moves through the food web.\nOptions: (A) kelp (B) kelp bass",
        'answer': "B",
        'lecture': 'A food web is a model.\nModels can make things in nature easier to understand. Models can be simpler than the things they represent. A food web is a model that shows where living things in an ecosystem get their food. If a food web showed every living thing in an ecosystem, the food web would be hard to understand. So, each food web shows how some living things in an ecosystem can get their food.\nArrows show how matter moves.\nA food web has arrows that point from one living thing to another. Each arrow shows the direction that matter moves when one living thing eats another living thing. An arrow starts from the living thing that is eaten. The arrow points to the living thing that is doing the eating.\nA living thing in a food web can have more than one arrow pointing from it. This shows that the living thing is eaten by more than one other living thing in the food web.\nA living thing in a food web can also have more than one arrow pointing to it. This shows that the living thing eats more than one other living thing in the food web.\nSOLUTION: Consumers eat other living things. So, there are arrows in a food web that point from other living things to consumers.\nThe kelp does not have any arrows pointing to it. So, the kelp is a producer, not a consumer.\nThe kelp bass has arrows pointing to it from the kelp, the zooplankton, and the plainfin midshipman. So, the kelp bass is a consumer.'
    },
    {
        'question': "What does the simile in this text suggest?\nTara rubbed coconut oil on her hands, which were like the parched earth during a drought.\nContext: N/A\nOptions: (A) Tara was baking something. (B) Tara's hands were dry and cracked.",
        'answer': "A",
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


        few_shot_prompt = ""

        for ex in examples:
            few_shot_prompt += f"{ex['question']} Think step by step and justify your steps. The answer is {ex['answer']} \n"
        # print(qs)
        # Concatenate the question and answer in a question-answer format
        prompt = few_shot_prompt + qs + " . Think step by step and justify your steps. "
        # prompt = few_shot_prompt + qs + " . Think step by step and justify your steps. "

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Move the inputs to the device
        inputs = inputs.to(device)

        # Generate a response
        outputs = model.generate(inputs, max_new_tokens = 128)

        # Decode the output tokens to text
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the original prompt from the output
        output_text = output_text.replace(prompt, '')
        prompt = prompt.replace(few_shot_prompt, '')
        
        # output_text = output_text.replace(qs, '')
        # output_text = output_text.replace(cur_prompt, '')
        # output_text = output_text.replace(". Think step by step and justify your steps.", '')

        # Add the generated response to the example
        # example['text'] = output_text

        # Write the example back to the JSONL file
        with open(args.output_path, 'a') as writer:
            #riter.write(json.dumps(output_text) + '\n')
            writer.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": output_text}) + "\n")

