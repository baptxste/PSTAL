import sys
import os
from collections import defaultdict
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
import numpy as np
import json

path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, '../pstal-etu/lib/'))
from conllulib import CoNLLUReader

tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')
model = AutoModel.from_pretrained('almanach/camembert-base')

def dataloader_with_embeddings(file):
    """
    Charge un fichier CoNLLU, extrait les embeddings des mots associés à leurs classes,
    et enregistre les résultats dans une structure.
    """
    data = []
    mapping = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(file, 'r') as file:
        reader = CoNLLUReader(file)
        for sent in reader.readConllu():
            mots = [tok["form"] for tok in sent]
            labels = [tok["frsemcor:noun"] for tok in sent]

            for label in labels:
                if label not in mapping:
                    mapping[label] = len(mapping)

            token_obj = tokenizer(mots, is_split_into_words=True, return_tensors='pt').to(device)
            word_ids = token_obj.word_ids()

            with torch.no_grad():
                embeddings = model(**token_obj)["last_hidden_state"]

            for word_idx in range(len(mots)):
                token_indices = [i for i, w_id in enumerate(word_ids) if w_id == word_idx]
                if token_indices:
                    avg_embedding = embeddings[:, token_indices, :].mean(dim=1).squeeze(0).cpu().numpy()
                    data.append({
                        "word": mots[word_idx],
                        "embedding": avg_embedding.tolist(),
                        "class": mapping[labels[word_idx]]
                    })

    return data, mapping

file_train = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train')
data_train, mapping = dataloader_with_embeddings(file_train)

with open("embeddings.json", "w") as file:
    json.dump(data_train, file, indent=4)

with open("mapping.json", "w") as file:
    json.dump(mapping, file, indent=4)
