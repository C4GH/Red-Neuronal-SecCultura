# Importación de bibliotecas estándar
import math
import time
import psutil

# Importación de bibliotecas relacionadas con PyTorch
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Módulo para agregar codificación posicional a las incrustaciones de tokens.

    Args:
    - emb_size (int): Tamaño de las incrustaciones.
    - dropout (float): Tasa de dropout a aplicar después de la codificación posicional.
    - maxlen (int): Longitud máxima de la secuencia que puede manejar el modelo.

    Attributes:
    - dropout (nn.Dropout): Capa de dropout.
    - pos_embedding (Tensor): Tensor de codificación posicional.
    """
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding.unsqueeze(-2))

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_joint_embedding.size(0), :])

class Modelo(nn.Module):
    """
    Modelo Transformer personalizado que integra incrustaciones, codificación posicional y un encoder.

    Args:
    - vocab_size (int): Tamaño del vocabulario.
    - embedding_dim (int): Dimensiones de las incrustaciones.
    - d_model (int): Dimensiones del modelo Transformer.
    - nhead (int): Número de cabezas de atención en cada bloque de atención.
    - num_layers (int): Número de capas del encoder Transformer.
    - num_tokens (int): Número de tokens de entrada.
    - num_classes (int): Número de clases de salida.
    - dropout (float): Tasa de dropout.

    Attributes:
    - embedding (nn.Embedding): Capa de incrustaciones.
    - positional_encoding (PositionalEncoding): Capa de codificación posicional.
    - encoder_layer (nn.TransformERencoderlayer): Capa única de encoder Transformer.
    - transformer_encoder (nn.TransformerEncoder): Encoder Transformer completo.
    - output_layer (nn.Linear): Capa lineal de salida.
    """
    def __init__(self, vocab_size, embedding_dim, d_model, nhead, num_layers, num_tokens, num_classes, dropout=0.05):
        super(Modelo, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = False
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src):
        """
        Define cómo pasa el input a través del modelo.

        Args:
        - src (Tensor): Tensor de índices de palabras.

        Returns:
        - Tensor: Predicciones del modelo.
        """
        src = self.embedding(src)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]  # Toma la salida del último token para la clasificación
        output = self.output_layer(output)
        return output
