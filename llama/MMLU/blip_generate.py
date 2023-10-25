import json
import jsonlines
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import requests
import torch
import pandas as pd
import csv
import torch
import os
import json
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import shutil

# Subcategories and categories mapping
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
    ("Which of these songs was a Top 10 hit for the rock band The Police?", "'Radio Ga-Ga'", "'Ob-la-di Ob-la-da'", "'In-a-Gadda-Da-Vida'", "'De Do Do Do De Da Da Da'",  "D"),
    # ("Which of the following is the best lens through which to investigate the role of child soldiers?", 
    #  "Child soldiers are victims of combat that need re-education and rehabilitation.", 
    #  "Children and their mothers are not active subjects in warfare and are best considered as subjects in the private sphere.", 
    #  "Children are most often innocent bystanders in war and are best used as signifiers of peace.", 
    #  "Children have political subjecthood that is missed when they are considered as passive victims of warfare.", 
    #  "D"),
    
    # ("At which stage in the planning process would a situation analysis be carried out?", 
    #  "Defining the program", 
    #  "Planning the program", 
    #  "Taking action and implementing ideas", 
    #  "Evaluation of the program", 
    #  "A"),
    
    # ("The realm of policy decisions concerned primarily with relations between the United States and the rest of the world is known as", 
    #  "terrorism policy.", 
    #  "economic policy.", 
    #  "foreign policy.", 
    #  "international policy.", 
    #  "C")
    #  ,
    
    # ("The term 'hegemony' refers to:", 
    #  "the tendency for the working class not to realize their own interests", 
    #  "a dominant ideology that legitimates economic, political and cultural power", 
    #  "a form of dual consciousness based on ideology and everyday experiences", 
    #  "a mode of payment given for outstanding topiary", 
    #  "B")
]

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the test folder')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder')
args = parser.parse_args()

# Load the InstructBLIP model and processor
model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path)
processor = InstructBlipProcessor.from_pretrained(args.model_path)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)  # replace 'model_name_or_path' with the name or path of your model


# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).to(dtype=torch.bfloat16)

# Use the blank image for every input
# Create a blank white image
width, height = 224, 224  # or any other desired size
blank_image = Image.new('RGB', (width, height), color='black')
raw_image = blank_image


few_shot_prompt = "\n\n".join([
    f"'{question}', Options: (A) {choiceA}, (B) {choiceB}, (C) {choiceC}, (D) {choiceD}, The correct answer is ({correct_answer}) {locals()[f'choice{correct_answer}']}."
    for question, choiceA, choiceB, choiceC, choiceD, correct_answer in data
])

# Create the "completed" folder if it doesn't exist
# completed_path = os.path.join(args.data_path, 'completed')
# os.makedirs(completed_path, exist_ok=True)


# Iterate through all CSV files in the test folder
for filename in os.listdir(args.data_path):
    if filename.endswith('.csv'):
        input_file_path = os.path.join(args.data_path, filename)
        subcategory = filename[:-9]  # Remove .csv extension
        category = [cat for cat, subcats in categories.items() if subcategories[subcategory][0] in subcats][0]

        # Create category folder if it doesn't exist
        category_path = os.path.join(args.output_path, category)
        os.makedirs(category_path, exist_ok=True)

        output_file_path = os.path.join(category_path, filename.replace('.csv', '_output.jsonl'))

        print(filename, flush = True)

        with open(input_file_path, 'r') as csvfile, open(output_file_path, 'w') as writer:
            reader = csv.reader(csvfile)
            for row in reader:
                stem = row[0]
                choices = [{'label': label, 'text': row[i+1]} for i, label in enumerate(['A', 'B', 'C', 'D'])]
                answerKey = row[5]

                # Format the choices text
                choices_text = ' '.join([f'({choice["label"]}) {choice["text"]}' for choice in choices])

                # Combine the stem and answer in a question-answer format
                prompt = f"'{stem}', Options: {choices_text}. The correct answer is "
                #prompt = few_shot_prompt + prompt
                # Tokenize the prompt
                # Process the blank image and the prompt
                #inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)

                # Assuming you have a text-only processor, process the prompt
                # inputs = processor(text=prompt, return_tensors="pt").to(device)
                # Process the blank image and the prompt
                truncated_tokens = tokenizer.encode(prompt, max_length=512, truncation=True)
                truncated_input = tokenizer.decode(truncated_tokens)
                inputs = processor(images=raw_image, text=truncated_input, return_tensors="pt").to(device).to(dtype=torch.bfloat16)

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

                #rationale = rationale.replace(few_shot_prompt, "")
                # Create the output example
                example = {
                    'question': stem,
                    'choices': choices,
                    'answerKey': answerKey,
                    'model_answer': rationale
                }

                # Write the example to the JSONL file
                with open(output_file_path, 'a') as writer:
                    writer.write(json.dumps(example) + '\n')
        # shutil.move(input_file_path, os.path.join(completed_path, filename))

