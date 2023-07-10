import json
import re

# correct_count = 0
# incorrect_count = 0
# invalid_count = 0
# label_counts = {"e": 0, "n": 0, "c": 0}
# correct_label_counts = {"e": 0, "n": 0, "c": 0}
# label_mapping = {"e": "entailment", "n": "neutral", "c": "contradiction"}

# def find_first_label_in_rationale(rationale, label_mapping):
#     words = re.findall(r"'(\w+)'|\w+", rationale.lower())  # extract words from the rationale including those in single quotes
#     for word in words:
#         for label, label_word in label_mapping.items():
#             if word == label_word.lower():
#                 return label  # return the first label that appears
#     return None  # if no labels appear, return None


# with open("anli_official_llama65B.jsonl", "r") as f:
#     for line in f:
#         item = json.loads(line)
#         rationale = item["model_answer"]
#         label = item["label"]
#         if not rationale:
#             invalid_count += 1
#             label_counts[label] += 1
#             continue
#         first_label_in_rationale = find_first_label_in_rationale(rationale, label_mapping)
#         label_word = label_mapping[label]  # this maps the label to its corresponding word
#         # Increase count for this label
#         label_counts[label] += 1

#         if first_label_in_rationale == label:
#             correct_count += 1
#             correct_label_counts[label] += 1  # Increase correct count for this label
#         elif first_label_in_rationale in label_mapping.keys():
#             incorrect_count += 1
#         else:
#             invalid_count += 1

# print(f"Correct labels: {correct_count}", flush = True)
# print(f"Incorrect label: {incorrect_count}", flush = True)
# print(f"Invalid label: {invalid_count}", flush = True)

# for label, count in label_counts.items():
#     correct_count_for_label = correct_label_counts[label]
#     percentage_correct = (correct_count_for_label / count) * 100 if count > 0 else 0  # calculate percentage
#     print(f"Occurrences of label '{label_mapping[label]}': {count}")
#     print(f"Correct predictions for label '{label_mapping[label]}': {correct_count_for_label} ({percentage_correct:.2f}%)")

import json
import re

correct_count = 0
incorrect_count = 0
invalid_count = 0

# Create a mapping from abbreviations to full label names
label_mapping = {"e": "entailment", "n": "neutral", "c": "contradiction"}

with open("anli_official_llama65B.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        model_answer = item["model_answer"]
        true_answer = label_mapping[item["label"]]  # Map abbreviation to full label name
        # Look for the pattern "'(entailment|contradiction|neutral)'" in the model_answer string
        match = re.search(r"'(entailment|contradiction|neutral)'", model_answer)
        if match:
            predicted_answer = match.group(1).strip()  # Extract the first matching group and remove any white spaces
            if predicted_answer == true_answer:
                correct_count += 1
            else:
                incorrect_count += 1
        else:
            invalid_count += 1

print(f"Correct answers: {correct_count}")
print(f"Incorrect answers: {incorrect_count}")
print(f"Invalid results: {invalid_count}")


