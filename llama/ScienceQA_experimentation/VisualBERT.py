import json
import torch
from transformers import AutoTokenizer, VisualBertForMultipleChoice
import argparse
from PIL import Image
import re


import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class VisualEmbeddingsExtractor:
    def __init__(self, device='cuda'):
        # Load the pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        
        # Remove the last fully connected layer to get the 2048-dimensional feature vector
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Set the model to evaluation mode and move to the desired device
        self.model.eval().to(device)
        
        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.device = device

    def get_visual_embeddings(self, image):
        # Apply the transformations and add an extra batch dimension
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get the embeddings
        with torch.no_grad():
            embeddings = self.model(image_tensor)
        
        # Remove the spatial dimensions
        embeddings = embeddings.squeeze(-1).squeeze(-1)
        
        return embeddings





# Create a black image of size 256x256
black_image = Image.new('RGB', (256, 256), color='black')

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to the jsonl data')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
parser.add_argument('--image_folder', type=str, required=True, help='Path to the images')

args = parser.parse_args()

# Load the VisualBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Open the JSON file
with open(args.data_path, 'r') as f:
    data = json.load(f)

# Open the output file
with open(args.output_path, 'a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(data):
        idx = example["id"]
        question = example['conversations'][0]
        qs = question['value']
        qs = qs.replace('<image>', '').strip()

        # Extract the answer options using regex
        options = re.findall(r'\(([A-E])\) (.*?)(?=\([A-E]\)|$|\n|<image>|\. )', qs)
        options = [option[1].strip() for option in options]

        if 'image' in example:
            image_file = example["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
        else:
            image = black_image

        # Assuming you have a way to get the image associated with the question
        # Usage
        embeddings_extractor = VisualEmbeddingsExtractor(device = device)
        visual_embeds = embeddings_extractor.get_visual_embeddings(image)
        visual_embeds = visual_embeds.expand(1, len(options), *visual_embeds.shape).to(device)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)

        # Tokenize the question and extracted choices
        encoding = tokenizer([[qs] * len(options), options], return_tensors="pt", padding=True)
        inputs_dict = {k: v.unsqueeze(0) for k, v in encoding.items()}
        inputs_dict.update(
            {
                "visual_embeds": visual_embeds,
                "visual_attention_mask": visual_attention_mask,
                "visual_token_type_ids": visual_token_type_ids,
            }
        )
        outputs = model(**inputs_dict)

        logits = outputs.logits
        # Get the predicted choice
        predicted_choice = torch.argmax(logits, dim=1).item()

        # Write the example back to the output file
        #writer.write(json.dumps({"question_id": idx, "predicted_choice": predicted_choice}) + "\n")
        # Write the example back to the JSONL file
        with open(args.output_path, 'a') as writer:
            #riter.write(json.dumps(output_text) + '\n')
            writer.write(json.dumps({"question_id": idx,
                                   "predicted_choice": predicted_choice}) + "\n")
