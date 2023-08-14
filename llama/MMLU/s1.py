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
category_counts = {category: {"tp": 0, "fp": 0, "fn": 0} for category in categories}

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

                # Compute sentence embedding for model answer
                model_answer_emb = model.encode([model_answer])[0]

                # Compute sentence embeddings for each choice and calculate cosine similarity
                similarities = []
                for choice in choices:
                    choice_emb = model.encode([choice["text"]])[0]
                    similarity = 1 - cosine(model_answer_emb, choice_emb)
                    similarities.append(similarity)

                # Find the choice with the highest similarity
                best_choice_index = similarities.index(max(similarities))
                best_choice = choices[best_choice_index]["label"]

                # Update counts for precision and recall
                if best_choice == true_answer:
                    category_counts[category]["tp"] += 1
                else:
                    category_counts[category]["fp"] += 1
                    category_counts[category]["fn"] += 1

# Compute S1 scores and print the results for each category
total_s1_score = 0
for category, counts in category_counts.items():
    precision = counts['tp'] / (counts['tp'] + counts['fp']) if (counts['tp'] + counts['fp']) > 0 else 0
    recall = counts['tp'] / (counts['tp'] + counts['fn']) if (counts['tp'] + counts['fn']) > 0 else 0
    
    if precision + recall > 0:
        s1_score = 2 * (precision * recall) / (precision + recall)
    else:
        s1_score = 0
    
    total_s1_score += s1_score
    s1_score = s1_score * 100
    print(f"{category}: S1 score {s1_score:.2f}", flush=True)
# Compute and print the average S1 score
average_s1_score = total_s1_score / len(categories)
average_s1_score = average_s1_score * 100
print(f"\nAverage S1 score across all categories: {average_s1_score:.2f}", flush=True)
print("",flush = True)