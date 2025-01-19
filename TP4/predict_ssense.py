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
from tqdm import tqdm

file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.test')
tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')
model = AutoModel.from_pretrained('almanach/camembert-base')



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



with open('mapping.json','r') as file :
    mapping = json.load(file)

mlp = torch.load("mlp.pth")
text = "Ouverture tous les jours sauf le lundi de 14h30 à 18h."
tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')
model = AutoModel.from_pretrained('almanach/camembert-base')



def test_model(model, tokenizer, input_sentence, map_labels):
    """
    Teste le modèle sur une phrase donnée et retourne la phrase avec les tags prédits.
    
    :param model: Le modèle entraîné (MLP).
    :param tokenizer: Le tokenizer utilisé pour le modèle de base.
    :param input_sentence: Une phrase en entrée (chaîne de caractères).
    :param map_labels: Un dictionnaire associant les indices aux étiquettes de classe.
    :return: Une liste de tuples (mot, tag_prédit).
    """
    # Détection de l'appareil
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Envoyer le modèle sur l'appareil
    model = model.to(device)

    # Tokenisation de la phrase
    token_obj = tokenizer(input_sentence.split(), is_split_into_words=True, return_tensors='pt')
    word_ids = token_obj.word_ids()
    
    with torch.no_grad():
        # Assurez-vous que les embeddings sont calculés sur le même appareil
        embeddings = AutoModel.from_pretrained('almanach/camembert-base').to(device)(**token_obj.to(device))["last_hidden_state"]

    # Calcul des embeddings moyens par mot
    word_embeddings = []
    for word_idx in range(len(input_sentence.split())):
        token_indices = [i for i, w_id in enumerate(word_ids) if w_id == word_idx]
        if token_indices:
            avg_embedding = embeddings[:, token_indices, :].mean(dim=1).squeeze(0)  # Moyenne des embeddings
            word_embeddings.append(avg_embedding)
        else:
            word_embeddings.append(torch.zeros(embeddings.size(-1), device=device))  # Embedding nul si aucun token

    # Conversion en tenseur pour le modèle
    word_embeddings = torch.stack(word_embeddings)

    # Prédictions
    model.eval()
    with torch.no_grad():
        predictions = model(word_embeddings.to(device))
        predicted_classes = torch.argmax(predictions, dim=1)

    # Association des mots avec leurs tags
    reverse_map = {v: k for k, v in map_labels.items()}
    tagged_sentence = [(word, reverse_map[predicted_classes[i].item()]) for i, word in enumerate(input_sentence.split())]

    return tagged_sentence


# Exemple d'utilisation



with open(os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.test'), 'r') as file:
    reader = CoNLLUReader(file)
    correct = 0
    total = 0
    for sent in reader.readConllu():
        mots = [tok["form"] for tok in sent]
        labels = [tok["frsemcor:noun"] for tok in sent]
        # text = sent['metadata']['text']
        # print(text)
        text = " ".join(mots)
        # print(f"text : {text}")
        # print(f"labels : {labels}")
        result = test_model(mlp, tokenizer, text, mapping)
        result = [e[1] for e in result]
        # print("Résultat annoté :", result)
        for i in range(len(labels)):
            if labels[i]!='*' : # labels != '*'
                total+=1
                if labels[i]==result[i]:
                    correct +=1
    print(correct/total)




# def dataloader(file=file_test):
#     """
#     Dataloader : charge un fichier CoNLLU, calcule des embeddings pour les mots
#     et associe les embeddings aux labels. Si un mot est décomposé en tokens,
#     l'embedding est la moyenne des embeddings des tokens du mot.
#     """
#     data = []
#     mapping = {}
#     with open(file, 'r') as file:
#         reader = CoNLLUReader(file)
#         for sent in reader.readConllu():
#             mots = [tok["form"] for tok in sent]
#             labels = [tok["frsemcor:noun"] for tok in sent]
#             label_idx = [i for i in range(len(labels))]
#             # print(mots)
#             # print(labels)
#             # print(label_idx)

#             for i in label_idx:
#                 if labels[i] not in mapping:
#                     mapping[labels[i]] = len(mapping)

#             token_obj = tokenizer(mots, is_split_into_words=True, return_tensors='pt')
#             word_ids = token_obj.word_ids()

#             with torch.no_grad():
#                 embeddings = model(**token_obj)["last_hidden_state"]

#             # calcul des embeddings moyens par mot
#             word_embeddings = []
#             for word_idx in range(len(mots)):
#                 token_indices = [i for i, w_id in enumerate(word_ids) if w_id == word_idx]
#                 if token_indices:
#                     avg_embedding = embeddings[:, token_indices, :].mean(dim=1).squeeze(0)
#                     word_embeddings.append(avg_embedding)
#                 else:
#                     word_embeddings.append(torch.zeros(embeddings.size(-1)).to(device)) 

#             # associer les embeddings moyens aux labels
#             for i in label_idx:
#                 if labels[i] in mapping: 
#                     data.append((word_embeddings[i], mapping[labels[i]]))

#     return data, mapping




# data_test, _ = dataloader(file_test)
# # for e in data_train[:50]:
# with open('mapping.json','r') as file : 
#     mapping = json.load(file)

# testloader = DataLoader(data_test, batch_size=10, shuffle=True)

# # print(f"Classes générées : {mapping}")
# # print(f"Nombre de classes : {len(mapping)}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = torch.load("mlp.pth").to(device)

# correct = 0
# for inputs, targets in testloader:
#     inputs, targets = inputs.to(device), targets.to(device)
#     pred = torch.argmax(model(inputs), dim=1)
#     if torch.equal(pred, targets) : 
#         correct+=1
# print(correct/len(testloader))

