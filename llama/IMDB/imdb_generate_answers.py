import json
import jsonlines
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import os

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_folder', type=str, required=True, help='Path to the folder containing JSON files')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
args = parser.parse_args()

# Load the LLMa model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, use_cache=True).cuda()

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

choices = "Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, Game-Show, History, Horror, Music, Musical, Mystery, News, Reality-TV, Romance, Sci-Fi, Sport, Talk-Show, Thriller, War, Western"

examples = [
    {
        'plot': 'A true story of a young woman whose abusive childhood results in her developing a multiple personality disorder.',
        'genres': 'Drama'
    },
    {
        'plot': 'In 1986, In Brooklyn, New York, the dysfunctional family of pseudo intellectuals composed by the university professor Bernard and the prominent writer Joan split. Bernard is a selfish, cheap and jealous decadent writer that rationalizes every attitude in his family and life and does not accept \"philistines\" - people that do not read books or watch movies, while the unfaithful Joan is growing as a writer and has no problems with \"philistines\". Their sons, the teenager Walt and the boy Frank, feel the separation and take side: Walt stays with Bernard, and Frank with Joan, and both are affected with abnormal behaviors. Frank drinks booze and smears with sperm the books in the library and a locker in the dress room of his school. The messed-up and insecure Walt uses Roger Water\'s song \"Hey You\" in a festival as if it was of his own, and breaks up with his girlfriend Sophie. Meanwhile Joan has an affair with Frank\'s tennis teacher Ivan and Bernard with his student Lili.',
        'genres': 'Comedy'
    },
    {
        'plot': 'Phil and Kate have a baby boy named Jake. They hire a baby-sitter, Camilla, to look after Jake and she becomes part of the family. The Sterling\'s friend and neighbor, Ned, takes a liking to Camilla and asks her out. She refuses, but Ned follows her and discovers that she is not quite human. Camilla discovers that she has been followed and Ned is pursued. He leaves a desperate message for Phil and Kate which reveals that Camilla has special plans for baby Jake.',
        'genres': 'Horror, Mystery, Thriller'
    },
    {
        'plot': 'During the psychedelic 60s and 70s Larry \"Doc\" Sportello is surprised by his former girlfriend and her plot for her billionaire boyfriend, his wife, and her boyfriend. A plan for kidnapping gets shaken up by the oddball characters entangled in this groovy kidnapping romp based upon the novel by Thomas Pynchon.',
        'genres': 'Crime, Mystery, Romance'
    }

]

few_shot_prompt = "\n\n".join([
    #f"Based on the plot description: '{example['plot']}', determine the most appropriate genres for this movie. Here are the choices: {choices}. The correct genres for this movie are: '{example['genres']}'."
    f"Based on the following plot description: '{plot}', which genres best describe this movie? List them as: Genres: '{example['genres']}'."
    for example in examples
])


# Loop through each JSON file in the specified folder
for json_file in os.listdir(args.data_folder):
    if json_file.endswith('.json'):
        with open(os.path.join(args.data_folder, json_file), 'r') as f:
            data = json.load(f)
            plot = data.get('plot', None)
            genres = data.get('genres', None)
            
            

            # Combine the stem and answer in a question-answer format
            #prompt = f"{few_shot_prompt} Based on the plot description: '{plot}', determine the most appropriate genres for this movie. Here are the choices: {choices}. The correct genres for this movie are: "
            prompt = f"{few_shot_prompt} Based on the following plot description: '{plot}', which genres best describe this movie? List them as: Genres: "


            # Tokenize the prompt
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            inputs = inputs.to(device)

            # Generate a response
            outputs = model.generate(inputs, max_new_tokens=50)

            # Decode the output tokens to text
            rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the few shot prompt from the rationale
            rationale = rationale.replace(prompt, "")

            # Create a dictionary containing the extracted information and the model's response
            output_data = {
                'plot': plot,
                'genres': genres,
                'model_response': rationale
            }

            # Write the example back to the JSONL file
            with open(args.output_path, 'a') as writer:
                writer.write(json.dumps(output_data) + '\n')
