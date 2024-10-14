import torch
import requests
from urllib.parse import quote
from torch.nn import functional as F
import os

# Assuming these are your model and tokenizer from your project
from modelo.ModeloC import Modelo
from embedding import Tokenizador

# Paths to necessary files
VOCOMPLETETEXT_PATH = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\Vocabulario\vocab_npa.npy"
MODELO_ARCH = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\modelo\Modelos entrenados\best_model_10Epochs.pth"
URL_BASE = 'https://sic.cultura.gob.mx/utiles/cosic/xcon.php?busquedaavanzada={cad}&p=1'

# Model parameters
vocab_size = 1866360
embedding_dim = 300
d_model = embedding_dim
nhead = 4
num_layers = 3
num_tokens = 300
num_classes = 3
dropout = 0.05

# Initialize tokenizer and model
print("Loading Vocabulary")
tokenizer = Tokenizador.Tokenizador(VOCOMPLETETEXT_PATH)
print("Loading Model")
model = Modelo(vocab_size, embedding_dim, d_model, nhead, num_layers, num_tokens, num_classes, dropout)
model.load_state_dict(torch.load(MODELO_ARCH))
model.eval()

# Disable gradient computation
torch.set_grad_enabled(False)

def clear_screen():
    """
    Clears the terminal screen.
    """
    # Check if the operating system is Windows
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def fetch_data(query):
    """
    Fetch data from the web and return JSON response and the URL for debugging.
    """
    encoded_query = quote(query)
    url = URL_BASE.format(cad=encoded_query)
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json(), url
        else:
            print(f'HTTP Error: {response.status_code}')
            print(response.text)
    except requests.RequestException as e:
        print(f'Error during requests to {url}: {str(e)}')
    return None, url

def prepara_datos(data, pregunta):
    """
    Prepare data for the model by concatenating names and context.
    Include a line jump before each 'nombre' for better readability.
    """
    text = ' '.join([f"\n{item['nombre']} {item['contexto']}" for item in data])
    return f'{pregunta} {text}'

def search_and_predict(query):
    data, url = fetch_data(query)
    if not data:
        print("No results found.")
        return

    processed_data = prepara_datos(data, query)
    tokens = tokenizer.vectoriza(processed_data)
    tokens = tokens.unsqueeze(0)  # Add batch dimension

    # Run model prediction
    logits = model(tokens)
    probabilities = F.softmax(logits, dim=1).squeeze().tolist()

    # Print fetched data with corresponding probabilities
    for idx, item in enumerate(data):
        print(f"Opción {idx + 1}: {item['nombre']} {item['contexto']} - Probabilidad: {probabilities[idx]:.4f}")

# Example usage
if __name__ == "__main__":
    while True:
        query = input("Enter search query or 'q' to quit: ")
        if query.lower() == 'q':
            break
        clear_screen()
        search_and_predict(query)
