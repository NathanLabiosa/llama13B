import json
import re
import argparse
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_most_similar_option(options, assistant_text):
    sentences = [assistant_text] + options
    embeddings = model.encode(sentences)
    similarities = [1 - cosine(embeddings[0], emb) for emb in embeddings[1:]]
    best_option_index = similarities.index(max(similarities))
    best_option = chr(ord('A') + best_option_index)
    return best_option

parser = argparse.ArgumentParser(description="Load data from a JSON file.")
parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON file to be loaded.")
parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to the ground truth JSONL file.")
args = parser.parse_args()

# Load the ground truth data from a JSONL file
ground_truth_data = []
with open(args.ground_truth_path, 'r') as f:
    for line in f:
        ground_truth_data.append(json.loads(line.strip()))

# Convert ground_truth_data into a dictionary
ground_truth_map = {item["question_id"]: item["ground_truth"] for item in ground_truth_data}

data = []
with open(args.data_path, 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Initialize counters
total_correct_predictions = 0
total_incorrect_predictions = 0

for item in data:
    ground_truth = ground_truth_map.get(item['question_id'], None)
    pred_text = item['text']
    # Split the text at "SOLUTION:" and take the part after it
    solution_parts = pred_text.split("\nSOLUTION:")
    if len(solution_parts) > 1:
        pred_text = solution_parts[1].strip()  # This will give you the text after "SOLUTION:"

    options = re.findall(r'\(([A-E])\) (.*?)(?=\([A-E]\)|$|\n|<image>|\. )', item['prompt'])
    options = [option[1].strip() for option in options]
    # print(options, flush = True)
    predicted_answer = get_most_similar_option(options, pred_text)
    if ground_truth == predicted_answer:
        total_correct_predictions += 1
    else:
        total_incorrect_predictions += 1

print(f"Total correct predictions: {total_correct_predictions}")
print(f"Total incorrect predictions: {total_incorrect_predictions}")
