import json
import re
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import scipy

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

# Load the data from the JSON file
with open('/home/nlabiosa/llama13B/llama/science_output.json') as f:
    data = json.load(f)

# Initialize counters
total_correct_predictions = 0
total_incorrect_predictions = 0

# Process "correct" and "incorrect" sections
for key in ["correct", "incorrect"]:
    for item in data[key]:
        ground_truth = item['ground_truth']
        pred_text = item['pred']

        # Extract the options from the question
        options = re.findall(r'\(([A-E])\) (.*?)(?=\([A-E]\)|$|\n|<image>)', item['question'])
        options = [option[1] for option in options]

        # Get most similar answer
        predicted_answer = get_most_similar_option(options, pred_text)

        if ground_truth == predicted_answer:
            total_correct_predictions += 1
        else:
            total_incorrect_predictions += 1
        #print(f"Total correct predictions: {total_correct_predictions}", flush = True)
        #print(f"Total incorrect predictions: {total_incorrect_predictions}", flush = True)
        #print(item['question_id'])
        

print(f"Total correct predictions: {total_correct_predictions}", flush = True)
print(f"Total incorrect predictions: {total_incorrect_predictions}", flush = True)
