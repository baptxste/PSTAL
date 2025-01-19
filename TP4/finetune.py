import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, AdamW
from transformers import get_scheduler
import sys
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, '../pstal-etu/lib/'))
from conllulib import CoNLLUReader
import json
from sklearn.metrics import classification_report


### LA classe automodelfortokenclasi... permet de deja avoir un model bert qui comprend un mlp en sortie 


# Paths to data
path = os.path.dirname(__file__)
file_train = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train')
file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
model = AutoModelForTokenClassification.from_pretrained("almanach/camembert-base", num_labels=10)  # Adjust num_labels

# Hyperparameters
batch_size = 16
num_epochs = 3
learning_rate = 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Custom Dataset
class CoNLLDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_mapping):
        self.data = []
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        with open(file_path, 'r') as file:
            reader = CoNLLUReader(file)
            for sent in reader.readConllu():
                tokens = [tok["form"] for tok in sent]
                labels = [label_mapping[tok["frsemcor:noun"]] for tok in sent]
                self.data.append((tokens, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]
        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True, padding="max_length", return_tensors="pt")
        label_ids = [-100] * len(encoding["input_ids"][0])  # Initialize with -100 (ignored by loss)
        word_ids = encoding.word_ids(0)
        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                label_ids[i] = labels[word_id]
        encoding["labels"] = torch.tensor(label_ids)
        return {key: val.squeeze(0) for key, val in encoding.items()}

# Prepare label mapping
def prepare_label_mapping(file):
    label_mapping = {}
    with open(file, 'r') as file:
        reader = CoNLLUReader(file)
        for sent in reader.readConllu():
            for tok in sent:
                label = tok["frsemcor:noun"]
                if label not in label_mapping:
                    label_mapping[label] = len(label_mapping)
    return label_mapping

label_mapping = prepare_label_mapping(file_train)
num_labels = len(label_mapping)

# Data loaders
train_dataset = CoNLLDataset(file_train, tokenizer, label_mapping)
test_dataset = CoNLLDataset(file_test, tokenizer, label_mapping)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_loader))

# Training loop
def train_model(model, train_loader, optimizer, scheduler, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
def evaluate_model(model, test_loader, device, label_mapping):
    model.eval()
    all_predictions = []
    all_labels = []
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)

            for pred, label, word_ids in zip(predictions, batch["labels"], batch["word_ids"]):
                pred_labels = [reverse_mapping[p.item()] for p, l in zip(pred, label) if l.item() != -100]
                true_labels = [reverse_mapping[l.item()] for l in label if l.item() != -100]
                all_predictions.extend(pred_labels)
                all_labels.extend(true_labels)

    print(classification_report(all_labels, all_predictions))

# Train and Evaluate
train_model(model, train_loader, optimizer, lr_scheduler, num_epochs, device)
evaluate_model(model, test_loader, device, label_mapping)

# Save the model and mapping
model.save_pretrained("camembert-finetuned")
tokenizer.save_pretrained("camembert-finetuned")
with open("label_mapping.json", "w") as f:
    json.dump(label_mapping, f)
