import json
import re

output_lines = []

with open("llama_anli_generate_fulltest.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        rationale = item["rationale"]

        # create the pattern to be removed
        removal_text = "Given the context: '{context}', the hypothesis: '{hypothesis}', it is a".format(context=item["context"], hypothesis=item["hypothesis"])
        
        # replace the pattern with an empty string
        new_rationale = rationale.replace(removal_text, "").strip()

        # update the rationale in the original item
        item["rationale"] = new_rationale

        # add the modified line to the output
        output_lines.append(json.dumps(item))

# write the modified lines to a new file
with open("clean_llama_anli_generate_fulltest.jsonl", "w") as f:
    for line in output_lines:
        f.write(line + "\n")
