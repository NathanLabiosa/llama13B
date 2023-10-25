import json
import re
import argparse

def get_answer_from_text(text):
    """
    Extract the answer (e.g., 'A', 'B', ...) from the model's output text.
    """
    # pattern = re.compile(r'answer is \(?[A-E]\)?')
    # pattern = re.compile(r'The answer is \(?\w*[A-E]\w*\)?')
    pattern = re.compile(r'\(?(\w*[A-E]\w*)\)?')


    matches = pattern.findall(text)
    return matches[0] if matches else None

parser = argparse.ArgumentParser(description="Load data from a JSON file.")
parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON file to be loaded.")
parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to the ground truth JSONL file.")
args = parser.parse_args()

# Load the ground truth data from a JSONL file
ground_truth_data = [json.loads(line.strip()) for line in open(args.ground_truth_path, 'r')]
ground_truth_map = {item["question_id"]: item["ground_truth"] for item in ground_truth_data}

# Initialize counters
total_correct_predictions = 0
total_incorrect_predictions = 0
total_correct_predictions_with_images = 0
total_correct_predictions_without_images = 0
none_type_count = 0

data = [json.loads(line.strip()) for line in open(args.data_path, 'r')]
print(args.data_path, flush=True)

for item in data:
    ground_truth = ground_truth_map.get(item['question_id'], None)
    predicted_answer = get_answer_from_text(item['text'])
    # print(type(ground_truth))
    # print(type(predicted_answer))
    if predicted_answer is None:
        none_type_count += 1

    if ground_truth == predicted_answer:
        total_correct_predictions += 1
        if 'image' in item:
            total_correct_predictions_with_images += 1
        else:
            total_correct_predictions_without_images += 1
    else:
        total_incorrect_predictions += 1

print(f"Total correct predictions: {total_correct_predictions}", flush=True)
print(f"    Total correct predictions with images: {total_correct_predictions_with_images}")
print(f"    Total correct predictions without images: {total_correct_predictions_without_images}")
print(f"Total incorrect predictions: {total_incorrect_predictions}", flush=True)
print(f"Total NoneType values encountered: {none_type_count}")
print("", flush=True)
