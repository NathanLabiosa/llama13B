import json
import os
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import argparse

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

categories = [
    "humanities",
    "STEM",
    "social sciences",
    "other (business, health, misc.)",
]

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to the test folder')
args = parser.parse_args()

base_path = args.data_path
print(args.data_path, flush=True)

# Dictionary to store the counts for each category
category_counts = {category: {"correct": 0, "incorrect": 0} for category in categories}

# Loop through the categories
for category in categories:
    category_path = os.path.join(base_path, category)
    
    # Loop through the result files within the category folder
    for result_file in os.listdir(category_path):
        result_file_path = os.path.join(category_path, result_file)
        
        with open(result_file_path, "r") as f:
            for line in f:
                item = json.loads(line)
                model_answer = item["model_answer"]
                true_answer = item["answerKey"]
                choices = item["choices"]
                question = item['question']

                
                choices_text = ', '.join([f'({choice["label"]}) {choice["text"]}' for choice in choices])

                # Combine the stem and answer in a question-answer format
                remove_string = f"Given the question: '{question}', the choices: {choices_text}, the correct answer is"
                model_answer = model_answer.replace(remove_string, "").strip()
                
                # Compute sentence embedding for model answer
                model_answer_emb = model.encode([model_answer])[0]

                # Compute sentence embeddings for each choice and calculate cosine similarity
                similarities = []
                for choice in choices:
                    choice_text_with_label = f"({choice['label']}) {choice['text']}"
                    choice_emb = model.encode([choice["text"]])[0]
                    similarity = 1 - cosine(model_answer_emb, choice_emb)
                    similarities.append(similarity)

                # Find the choice with the highest similarity
                best_choice_index = similarities.index(max(similarities))
                best_choice = choices[best_choice_index]["label"]

                # Check if the best choice matches the correct answer
                if best_choice == true_answer:
                    category_counts[category]["correct"] += 1
                else:
                    category_counts[category]["incorrect"] += 1

total_accuracy = 0
num_categories = 0
# Print the results for each category
for category, counts in category_counts.items():
    total = counts['correct'] + counts['incorrect']
    if total > 0:
        accuracy = (counts['correct'] / total) * 100
        total_accuracy += accuracy
        num_categories += 1
    else:
        accuracy = 0
    print(f"{category}: correct {counts['correct']} incorrect {counts['incorrect']} accuracy {accuracy:.2f}%", flush=True)

if num_categories > 0:
    average_accuracy = total_accuracy / num_categories
else:
    average_accuracy = 0

print(f"Average accuracy across all categories: {average_accuracy:.2f}%", flush=True)
