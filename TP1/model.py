import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size_w, vocab_size_t, embedding_dim, hidden_dim, dropout_rate, pad_id):
        """
        Constructeur du modèle RNN d'étiquetage.

        :param vocab_size_w: Taille du vocabulaire des mots (|Vw|)
        :param vocab_size_t: Taille du vocabulaire des étiquettes (|Vt|)
        :param embedding_dim: Dimension des embeddings des mots
        :param hidden_dim: Dimension du vecteur caché de la couche GRU
        :param dropout_rate: Taux de dropout pour la régularisation
        :param pad_id: ID du padding, utilisé pour la couche d'embedding (padding_idx)
        """
        super(Model, self).__init__()

        self.embedding = nn.Embedding(vocab_size_w, embedding_dim, padding_idx=pad_id)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, bias=False)

       # Dropout pour régularisation
        self.dropout = nn.Dropout(dropout_rate)

        # Linear Layer (pour la classification des étiquettes)
        # La couche linéaire prend en entrée un vecteur caché de dimension 2*hidden_dim (bidirectionnel) et renvoie une probabilité sur |Vt| étiquettes.
        self.fc = nn.Linear(2 * hidden_dim, vocab_size_t)

    def forward(self, x):
        """
        Propagation avant du modèle.

        :param x: Tenseur d'indices de mots, de forme (B, L), où B est la taille du batch et L est la longueur de la phrase.
        :return: Tenseur de scores pour chaque étiquette de forme (B, L, |Vt|)
        """
        embedded = self.embedding(x)

        # Passage à travers la couche GRU
        # output : (B, L, 2*hidden_dim) => une sortie pour chaque mot de la phrase
        # hidden : (2, B, hidden_dim) => états cachés de la première et de la deuxième direction
        gru_out, _ = self.gru(embedded)

        gru_out = self.dropout(gru_out)

        # Passer les sorties du GRU à la couche linéaire pour obtenir les scores pour chaque étiquette
        out = self.fc(gru_out)  # (B, L, |Vt|)

        return out

# # Exemple d'initialisation du modèle
# vocab_size_w = len(vocab_w)  # Taille du vocabulaire des mots
# vocab_size_t = len(vocab_t)  # Taille du vocabulaire des étiquettes
# embedding_dim = 100  # Dimension des embeddings des mots
# hidden_dim = 128  # Dimension des états cachés
# dropout_rate = 0.5  # Taux de dropout
# pad_id = vocab_w["<PAD>"]  # ID du padding

# model = Model(vocab_size_w, vocab_size_t, embedding_dim, hidden_dim, dropout_rate, pad_id)

# # Affichage de la structure du modèle
# print(model)
