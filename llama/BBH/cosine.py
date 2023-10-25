import os
import json
import re
import argparse
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

total_correct_predictions = 0
total_incorrect_predictions = 0

def get_most_similar_option(options, assistant_text):
    sentences = [assistant_text] + options
    embeddings = model.encode(sentences)
    similarities = [1 - cosine(embeddings[0], emb) for emb in embeddings[1:]]
    best_option_index = similarities.index(max(similarities))
    best_option = f"({chr(ord('A') + best_option_index)})"
    return best_option

def process_true_false_data(item):
    """
    Process the true/false data format.
    """
    global total_correct_predictions, total_incorrect_predictions
    ground_truth = item['answerKey']
    pred_text = item['model_answer']
    
    # Extract the words "True" and "False" from the model's answer
    predicted_answer = "True" if "True" in pred_text else "False" if "False" in pred_text else None
    
    # Compare the predicted answer with the ground truth
    if ground_truth == predicted_answer:
        total_correct_predictions += 1
    else:
        total_incorrect_predictions += 1

def process_options_data(item):
    """
    Process the options data format.
    """
    global total_correct_predictions, total_incorrect_predictions
    ground_truth = item['answerKey']
    pred_text = item['model_answer']
    options = item['options']
    
    
    predicted_answer = get_most_similar_option(options, pred_text)
    

    
    # Debugging prints
    # print(f"Model's Answer: {pred_text}")
    # print(f"Predicted Option: {predicted_answer}")
    # print(f"Ground Truth: {ground_truth}")
    # print("------")
    # Compare the predicted answer with the ground truth
    if ground_truth == predicted_answer:
        total_correct_predictions += 1
    else:
        total_incorrect_predictions += 1


parser = argparse.ArgumentParser(description="Load data from a JSONL file.")
parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing JSONL files.")
args = parser.parse_args()
print(args.data_folder)

# Loop over all JSONL files in the data folder
for filename in os.listdir(args.data_folder):
    if filename.endswith('.jsonl'):
        with open(os.path.join(args.data_folder, filename), 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        # Check if any item in the file has the "options" key
        if any("options" in item for item in data):
            print(filename)
            for item in data:
                # Determine the type of data format and call the appropriate function
                if "answerKey" in item and isinstance(item["answerKey"], bool):
                    process_true_false_data(item)
                elif "options" in item:
                    process_options_data(item)

total_predictions = total_correct_predictions + total_incorrect_predictions
accuracy = total_correct_predictions / total_predictions * 100

print(f"Total correct predictions: {total_correct_predictions}")
print(f"Total incorrect predictions: {total_incorrect_predictions}")
print(f"Total predictions: {total_predictions}")
print(f"Accuracy: {accuracy:.2f}%")