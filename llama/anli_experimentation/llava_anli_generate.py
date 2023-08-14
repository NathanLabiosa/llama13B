import argparse
import os
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llava.conversation import default_conversation
from llava.utils import disable_torch_init

@torch.inference_mode()
def eval_model(model_name, input_file, output_file):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

    # Few-shot learning examples and label mapping
    label_mapping = {"e": "entailment", "n": "neutral", "c": "contradiction"}
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
    few_shot_prompt = "\n\n".join([
        f"Given the context: '{example['context']}', the hypothesis: '{example['hypothesis']}', what is their relationship? Think step by step and justify your answer. '{examples['rationale']}' Therefore the relationship is'{label_mapping[example['label']]}'. "
        for example in examples
    ])

    with jsonlines.open(os.path.expanduser(input_file), mode='r') as reader, jsonlines.open(os.path.expanduser(output_file), mode='w') as writer:
        for i, example in enumerate(reader):
            context = example["context"]
            hypothesis = example["hypothesis"]
            answer = label_mapping[example["label"]]

            # Combine the context, hypothesis, and answer in a question-answer format
            prompt = f"{few_shot_prompt}\n\nGiven the context: '{context}', the hypothesis: '{hypothesis}', what is their relationship?  Think step by step and justify your answer. "

            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

            # Generate a response
            outputs = model.generate(inputs, max_new_tokens=150)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the few-shot examples from the output
            output_text = output_text.replace(few_shot_prompt, '')

            # Add the rationale to the example
            example["rationale"] = output_text

            writer.write(example)

            # Print a status update
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} examples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/home/nlabiosa/LLaVA/llava13B")  # replace with the path to your model
    parser.add_argument("--input-file", type=str, default="anli/data/anli_v1.0/R1/minitest.jsonl")
    parser.add_argument("--output-file", type=str, default="llava_anli_generate.jsonl")
    args = parser.parse_args()

    eval_model(args.model_name, args.input_file, args.output_file)
