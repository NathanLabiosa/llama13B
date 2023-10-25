import json
import re
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import scipy
import argparse

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_most_similar_option(options, assistant_text):
    sentences = [assistant_text] + options

    # Calculate the sentence embeddings
    embeddings = model.encode(sentences)

    # Calculate similarities
    similarities = []
    for i in range(1, len(options) + 1):
        similarities.append(1 - cosine(embeddings[0], embeddings[i]))

    # Find the option with the highest similarity
    best_option_index = similarities.index(max(similarities))
    best_option = chr(ord('A') + best_option_index)

    return best_option


parser = argparse.ArgumentParser(description="Load data from a JSON file.")

# Add argument for data path
parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the JSON file to be loaded.")
parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to the ground truth JSONL file.")
                        
    
args = parser.parse_args()
# Load the data from the JSON file
#with open('/home/nlabiosa/llama13B/llama/ScienceQA_experimentation/blip_noimage_output.json') as f:
    # data = json.load(f)

# Load the ground truth data from a JSONL file
ground_truth_data = []
with open(args.ground_truth_path, 'r') as f:
    for line in f:
        ground_truth_data.append(json.loads(line.strip()))

# Convert ground_truth_data into a dictionary
ground_truth_map = {item["question_id"]: item["ground_truth"] for item in ground_truth_data}


def load_options_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    options_map = {}
    for item in data:
        question_id = item["id"]
        for conversation in item["conversations"]:
            if conversation["from"] == "human":
                value = conversation["value"]
                options_str = value.split("Options: ")[-1]
                options = re.findall(r'\([A-E]\) [^\(]*', options_str)
                options_map[question_id] = options
                break

    return options_map

options_map = load_options_from_file('/home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json')



# Initialize counters
total_correct_predictions = 0
total_incorrect_predictions = 0

# Initialize counters
total_correct_predictions_with_images = 0
total_correct_predictions_without_images = 0

data = []
with open(args.data_path, 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

print(args.data_path, flush=True)
for item in data:
    ground_truth = ground_truth_map.get(item['question_id'], None)  # Use the mapping to get the ground truth
    pred_text = item['text']
    question_id = item['question_id']

    # Extract the options from the question
    # options = re.findall(r'\(([A-E])\) (.*?)(?=\([A-E]\)|$|\n|<image>)', item['prompt'])
    # options = re.findall(r'\(([A-E])\) (.*?)(?=\([A-E]\)|$|\n|<image>|\. )', item['prompt'])
    # print(options, flush=True)
    # options = [f"({option[0]}) {option[1]}" for option in options]  # Include the option label
    options = options_map[question_id]
    # print(options)

    # Get most similar answer
    # print(options, flush = True)
    predicted_answer = get_most_similar_option(options, pred_text)
    
    if ground_truth == predicted_answer:
        total_correct_predictions += 1
        if 'image' in item:
            total_correct_predictions_with_images += 1
        else:
            total_correct_predictions_without_images += 1
    else:
        total_incorrect_predictions += 1
    #print(f"Total correct predictions: {total_correct_predictions}", flush = True)
    #print(f"Total incorrect predictions: {total_incorrect_predictions}", flush = True)
    #print(item['question_id'])
        

print(f"Total correct predictions: {total_correct_predictions}", flush = True)
print(f"    Total correct predictions with images: {total_correct_predictions_with_images}")
print(f"    Total correct predictions without images: {total_correct_predictions_without_images}")
print(f"Total incorrect predictions: {total_incorrect_predictions}", flush = True)
print("", flush = True)
