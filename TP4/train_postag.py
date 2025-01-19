import sys
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from matplotlib import pyplot as plt
import json

path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, '../pstal-etu/lib/'))
from conllulib import Util, CoNLLUReader

tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')
model = AutoModel.from_pretrained('almanach/camembert-base')

file_train = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train')
file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev')



def dataloader(file=file_test):
    """
    Dataloader : charge un fichier CoNLLU, calcule des embeddings pour les mots
    et associe les embeddings aux labels. Si un mot est décomposé en tokens,
    l'embedding est la moyenne des embeddings des tokens du mot.
    """
    data = []
    mapping = {}
    with open(file, 'r') as file:
        reader = CoNLLUReader(file)
        for sent in reader.readConllu():
            mots = [tok["form"] for tok in sent]
            labels = [tok["frsemcor:noun"] for tok in sent]
            label_idx = [i for i in range(len(labels))]
            # print(mots)
            # print(labels)
            # print(label_idx)

            for i in label_idx:
                if labels[i] not in mapping:
                    mapping[labels[i]] = len(mapping)

            token_obj = tokenizer(mots, is_split_into_words=True, return_tensors='pt')
            word_ids = token_obj.word_ids()

            with torch.no_grad():
                embeddings = model(**token_obj)["last_hidden_state"]

            # calcul des embeddings moyens par mot
            word_embeddings = []
            for word_idx in range(len(mots)):
                token_indices = [i for i, w_id in enumerate(word_ids) if w_id == word_idx]
                if token_indices:
                    avg_embedding = embeddings[:, token_indices, :].mean(dim=1).squeeze(0)
                    word_embeddings.append(avg_embedding)
                else:
                    word_embeddings.append(torch.zeros(embeddings.size(-1)).to(device)) 

            # associer les embeddings moyens aux labels
            for i in label_idx:
                if labels[i] in mapping: 
                    data.append((word_embeddings[i], mapping[labels[i]]))

    return data, mapping


class MLP(nn.Module):
    def __init__(self, embedding_size, h1, h2, nb_tags):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=embedding_size, out_features=h1)
        self.fc2 = nn.Linear(in_features=h1, out_features=h2)
        self.fc3 = nn.Linear(in_features=h2, out_features=nb_tags)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x





def train_with_metrics(model_nn, trainloader, validateloader, loss_fn, n_epochs, optimizer):
    """
    Fonction d'entraînement du modèle avec calcul d'accuracy.
    :param model_nn: modèle de type nn.Module
    :param trainloader: DataLoader pour l'entraînement
    :param validateloader: DataLoader pour la validation
    :param loss_fn: fonction de perte
    :param n_epochs: nombre d'époques
    :param optimizer: optimiseur
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_nn = model_nn.to(device)
    losses = []
    accuracies = []

    for epoch in range(n_epochs):
        # Phase d'entraînement
        model_nn.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model_nn(inputs)

            # Calcul de la perte
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calcul de l'accuracy pour ce batch
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        train_accuracy = correct_train / total_train

        # Phase de validation
        model_nn.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in validateloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_nn(inputs)

                # Calcul de la perte
                val_loss += loss_fn(outputs, targets).item()

                # Calcul de l'accuracy pour ce batch
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        val_accuracy = correct_val / total_val

        # Moyennes pour l'époque
        train_loss /= len(trainloader)
        val_loss /= len(validateloader)
        losses.append(val_loss)
        accuracies.append(val_accuracy)

        print(f"Époque {epoch + 1}/{n_epochs} : "
              f"Perte entraînement = {train_loss:.4f}, "
              f"Accuracy entraînement = {train_accuracy:.4f}, "
              f"Perte validation = {val_loss:.4f}, "
              f"Accuracy validation = {val_accuracy:.4f}")

    print("Entraînement terminé.")
    return losses, accuracies


data_train, mapping = dataloader(file_test)
# for e in data_train[:50]:
    # print(e[1])
data_eval, _ = dataloader(file_test)

trainloader = DataLoader(data_train, batch_size=10, shuffle=True)
evalloader = DataLoader(data_eval, batch_size=10, shuffle=True)

print(f"Classes générées : {mapping}")
print(f"Nombre de classes : {len(mapping)}")

model = MLP(768, 200, 200, len(mapping))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()


print("Entraînement")
# losses = train(model, trainloader, evalloader, loss, 20, optimizer)
# torch.save(model, "mlp.pth")
# mapping = dict(mapping)
# with open("map.json","w") as file : 
#     json.dump(mapping, file, indent=4)
# plt.plot(losses)
# plt.title("Courbe des pertes de validation")
# plt.xlabel("Époques")
# plt.ylabel("Perte")
# plt.show()


losses, accuracies = train_with_metrics(model, trainloader, evalloader, loss, 20, optimizer)
torch.save(model, "mlp.pth")
mapping = dict(mapping)
with open("mapping.json", "w") as file: 
    json.dump(mapping, file, indent=4)

# Tracer la courbe des pertes
plt.plot(losses, label="Perte validation")
plt.title("Courbe des pertes de validation")
plt.xlabel("Époques")
plt.ylabel("Perte")
plt.legend()
plt.show()

# Tracer la courbe des accuracies
plt.plot(accuracies, label="Accuracy validation")
plt.title("Courbe des accuracies de validation")
plt.xlabel("Époques")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

