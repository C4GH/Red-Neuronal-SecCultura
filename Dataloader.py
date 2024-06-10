# Standard libraries
import json
import time

# Third-Party libraries
import numpy as np

# Torch-related libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, json_path, embeddings_path, vocab_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float)
        vocab_words = np.load(vocab_path)
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
        if '<unk>' not in self.vocab:
            raise ValueError("El token '<unk>' debe de estar en el vocabulario")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item[0]
        indices = [self.vocab.get(word, self.vocab.get('<unk>')) for word in text.split()]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        label = torch.tensor(item[1]).argmax().item()  # Convert label form probability list to class index
        return indices_tensor, label

def collate_fn(batch):
    indices_list, labels_list = zip(*batch)
    padded_indices = pad_sequence(indices_list, batch_first=True, padding_value=0)
    desired_length = 300
    if padded_indices.shape[1] > desired_length:
        padded_indices = padded_indices[:, :desired_length]
    elif padded_indices.shape[1] < desired_length:
        padding_size = desired_length - padded_indices.shape[1]
        padded_indices = torch.cat([padded_indices, torch.zeros(len(padded_indices), padding_size, dtype=torch.long)], dim=1)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    return padded_indices, labels_tensor

if __name__ == '__main__':
    json_path = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\Vocabulario\salida_min.json"
    embeddings_path = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\Vocabulario\embs_npa.npy"
    vocab_path = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\Vocabulario\vocab_npa.npy"
    dataset = CustomDataset(json_path, embeddings_path, vocab_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
