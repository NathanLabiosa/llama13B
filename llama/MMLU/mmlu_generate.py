import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import csv
from llava.conversation import default_conversation
from llava.utils import disable_torch_init

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
} 


data = [
    ("What place is named in the title of the 1979 live album by rock legends Cheap Trick?", "Budapest", "Budokan", "Bhutan", "Britain", "B"),
    ("Who is the shortest man to ever win an NBA slam dunk competition?", "Anthony 'Spud' Webb", "Michael 'Air' Jordan", "Tyrone 'Muggsy' Bogues", "Julius 'Dr J' Erving", "A"),
    ("What is produced during photosynthesis?", "hydrogen", "nylon", "oxygen", "light", "C"),
    ("Which of these songs was a Top 10 hit for the rock band The Police?", "'Radio Ga-Ga'", "'Ob-la-di Ob-la-da'", "'In-a-Gadda-Da-Vida'", "'De Do Do Do De Da Da Da'",  "D")
    ]

few_shot_prompt = "\n\n".join([
    f"'{question}', Options: (A) {choiceA}, (B) {choiceB}, (C) {choiceC}, (D) {choiceD}, The correct answer is ({correct_answer}) {locals()[f'choice{correct_answer}']}."
    for question, choiceA, choiceB, choiceC, choiceD, correct_answer in data
])

# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


@torch.inference_mode()
def eval_model(model_name, data_path, output_path):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16).cuda()

    for filename in os.listdir(args.data_path):
        if filename.endswith('.csv'):
            input_file_path = os.path.join(args.data_path, filename)
            subcategory = filename[:-9]  # Remove .csv extension
            category = [cat for cat, subcats in categories.items() if subcategories[subcategory][0] in subcats][0]

            # Create category folder if it doesn't exist
            category_path = os.path.join(args.output_path, category)
            os.makedirs(category_path, exist_ok=True)

            output_file_path = os.path.join(category_path, filename.replace('.csv', '_output.jsonl'))

            with open(input_file_path, 'r') as csvfile, open(output_file_path, 'w') as writer:
                reader = csv.reader(csvfile)
                for row in reader:
                    stem = row[0]
                    choices = [{'label': label, 'text': row[i+1]} for i, label in enumerate(['A', 'B', 'C', 'D'])]
                    answerKey = row[5]

                    # Format the choices text
                    choices_text = ', '.join([f'({choice["label"]}) {choice["text"]}' for choice in choices])

                    # Combine the stem and answer in a question-answer format
                    prompt = f"Given the question: '{stem}', the choices: {choices_text}, the correct answer is"
                    few_shot_prompt = ""
                    prompt = few_shot_prompt + prompt
                    qs = prompt



                    conv = default_conversation.copy()
                    conv.append_message(conv.roles[0], qs)
                    prompt = conv.get_prompt()
                    inputs = tokenizer([prompt])
                    input_ids = torch.as_tensor(inputs.input_ids).cuda()
                    stopping_criteria = KeywordsStoppingCriteria([conv.sep], tokenizer, input_ids)
                    output_ids = model.generate(
                        input_ids,
                        do_sample=True,
                        use_cache=True,
                        temperature=0.7,
                        max_new_tokens=1024,
                        stopping_criteria=[stopping_criteria])
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    try:
                        index = outputs.index(conv.sep, len(prompt))
                    except ValueError:
                        outputs += conv.sep
                        index = outputs.index(conv.sep, len(prompt))

                    outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()
                    outputs = outputs.replace(few_shot_prompt, "")
                    example = {
                    'question': stem,
                    'choices': choices,
                    'answerKey': answerKey,
                    'model_answer': outputs
                    }
                    # Write the example to the JSONL file
                    with open(output_file_path, 'a') as writer:
                        writer.write(json.dumps(example) + '\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the test folder')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder')
    args = parser.parse_args()

    eval_model(args.model_name, args.data_path, args.output_path)
