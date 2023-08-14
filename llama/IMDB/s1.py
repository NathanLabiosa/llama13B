import json
import os
from sentence_transformers import SentenceTransformer
import argparse

# Load the sentence transformer model
#model = SentenceTransformer('all-MiniLM-L6-v2')

# List of all possible genres
all_genres = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", 
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "Game-Show", 
    "History", "Horror", "Music", "Musical", "Mystery", "News", "Reality-TV", 
    "Romance", "Sci-Fi", "Sport", "Talk-Show", "Thriller", "War", "Western"
]


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to the test folder')
args = parser.parse_args()

base_path = args.data_path
print(args.data_path, flush=True)

# Dictionary to store the counts for each genre
genre_counts = {genre: {"tp": 0, "fp": 0, "fn": 0} for genre in all_genres}

# Loop through the data for LLaMA
# with open(base_path, "r") as f:
#     for line in f:
#         item = json.loads(line)
#         model_response = item.get("model_response", "").strip()  # Get model response and strip any whitespace

#         # Check if the model_response is not empty and contains the expected format
#         if model_response and "'" in model_response:
#             model_genres = model_response.split("'")[1].split(", ")  # Extract genres from model's response
#         else:
#             model_genres = []  # If the response is blank or unexpected, set model_genres to an empty list

#         true_genres = item.get("genres", [])

#         for genre in all_genres:
#             if genre in model_genres and genre in true_genres:
#                 genre_counts[genre]["tp"] += 1
#             elif genre in model_genres:
#                 genre_counts[genre]["fp"] += 1
#             elif genre in true_genres:
#                 genre_counts[genre]["fn"] += 1

# Loop through the data for Vicuna
# with open(base_path, "r") as f:
#     for line in f:
#         item = json.loads(line)
#         model_response = item.get("model_response", "").strip()  # Get model response and strip any whitespace

#         if model_response:
#             # Split by comma and then extract genre after the period
#             model_genres = [genre.split('.')[-1].strip() for genre in model_response.split(',')]
#         else:
#             model_genres = []  # If the response is blank, set model_genres to an empty list

#         true_genres = item.get("genres", [])

#         for genre in all_genres:
#             if genre in model_genres and genre in true_genres:
#                 genre_counts[genre]["tp"] += 1
#             elif genre in model_genres:
#                 genre_counts[genre]["fp"] += 1
#             elif genre in true_genres:
#                 genre_counts[genre]["fn"] += 1
# Loop through the data for LLaVA
import re

with open(base_path, "r") as f:
    for line in f:
        item = json.loads(line)
        model_response = item.get("model_response", "").strip().lower()  # Get model response, strip any whitespace, and convert to lowercase

        true_genres = [genre.lower() for genre in item.get("genres", [])]  # Convert true genres to lowercase

        for genre in all_genres:
            genre_lower = genre.lower()  # Convert the genre to lowercase for comparison
            # Use regex to check if the genre is present as a whole word in the model_response
            if re.search(r'\b' + re.escape(genre_lower) + r'\b', model_response) and genre_lower in true_genres:
                genre_counts[genre]["tp"] += 1
            elif re.search(r'\b' + re.escape(genre_lower) + r'\b', model_response):
                genre_counts[genre]["fp"] += 1
            elif genre_lower in true_genres:
                genre_counts[genre]["fn"] += 1



# Compute S1 scores and print the results for each genre
total_s1_score = 0
for genre, counts in genre_counts.items():
    precision = counts['tp'] / (counts['tp'] + counts['fp']) if (counts['tp'] + counts['fp']) > 0 else 0
    recall = counts['tp'] / (counts['tp'] + counts['fn']) if (counts['tp'] + counts['fn']) > 0 else 0
    
    if precision + recall > 0:
        s1_score = 2 * (precision * recall) / (precision + recall)
    else:
        s1_score = 0
    
    total_s1_score += s1_score
    s1_score = s1_score * 100
    print(f"{genre}: S1 score {s1_score:.2f}", flush=True)

# Compute and print the average S1 score
average_s1_score = total_s1_score / len(all_genres)
average_s1_score = average_s1_score * 100
print(f"\nAverage S1 score across all genres: {average_s1_score:.2f}", flush=True)
print("", flush=True)
