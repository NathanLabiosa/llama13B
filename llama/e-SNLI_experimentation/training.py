import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW

class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Load your data
df = pd.read_csv('esnli_minitest.csv')

# Convert text labels to numbers
label_to_id = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
df['label'] = df['gold_label'].map(label_to_id)

# Combine the sentences into one input string
df['input'] = "Sentence 1: " + df['Sentence1'] + " Sentence 2: " + df['Sentence2']

# Load the LLMa model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained('/home/nlabiosa/llama13B/llama/llama-7b-hf')

# Add a padding token to the tokenizer
if tokenizer._pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = LlamaForCausalLM.from_pretrained('/home/nlabiosa/llama13B/llama/llama-7b-hf', torch_dtype=torch.float16).to('cuda')

# Resize the token embeddings
model.resize_token_embeddings(len(tokenizer))

# Tokenize the inputs
inputs = tokenizer(df['input'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)

# Move inputs to GPU
inputs = {k: v.to('cuda') for k, v in inputs.items()}

# Create PyTorch Dataset and DataLoader
dataset = MyDataset(inputs, df['label'].tolist())
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Initialize the optimizer
optim = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # You can change the number of epochs
    for batch in dataloader:

        # Forward pass
        outputs = model(**batch)

        # Backward pass
        loss = outputs.loss
        loss.backward()

        # Update weights
        optim.step()
        optim.zero_grad()

# Save the fine-tuned model
model.save_pretrained('fine-tuned-model')
