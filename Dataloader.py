# Importación de bibliotecas estándar
import json
import time

# Importación de bibliotecas de terceros
import numpy as np

# Importación de bibliotecas relacionadas con PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    """
    Clase para crear un conjunto de datos personalizado que hereda de `Dataset`.

    Args:
    - json_path (str): Ruta al archivo JSON con los datos.
    - embeddings_path (str): Ruta al archivo Numpy con los embeddings.
    - vocab_path (str): Ruta al archivo Numpy con el vocabulario.

    Attributes:
    - data (list): Datos extraídos del archivo JSON.
    - embeddings (Tensor): Embeddings de palabras como tensores de PyTorch.
    - vocab (dict): Diccionario que mapea palabras a índices enteros.
    """
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
        label = torch.tensor(item[1]).argmax().item()  # Convertir la lista de probabilidades a índice de clase
        return indices_tensor, label

def collate_fn(batch):
    """
    Función para manejar cómo se agrupan las muestras en un lote.

    Args:
    - batch (list): Lista de tuplas (indices_tensor, label).

    Returns:
    - Tensor de índices con padding y Tensor de etiquetas.
    """
    indices_list, labels_list = zip(*batch)
    padded_indices = pad_sequence(indices_list, batch_first=True, padding_value=0)
    desired_length = 300
    if padded_indices.shape[1] > desired_array:
        padded_indices = padded_indices[:, :desired_array]
    elif padded_indices.shape[1] < desired_length:
        padding_size = desired_length - padded_indices.shape[1]
        padded_indices = torch.cat([padded_indices, torch.zeros(len(padded_indices), padding_size, dtype=torch.long)], dim=1)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    return padded_indices, labels_tensor

if __name__ == '__main__':
    # Inicialización de dataset y dataloader con rutas de archivos sensibles no incluidas
    dataset = CustomDataset('<json_path>', '<embeddings_path>', '<vocab_path>')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
