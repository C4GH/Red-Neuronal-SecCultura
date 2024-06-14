import torch
import requests  # Utilizada para realizar peticiones HTTP
from urllib.parse import quote  # Función para codificar URL
from torch.nn import functional as F  # Contiene funciones de redes neuronales como softmax
import os  # Funciones del sistema operativo para limpiar la pantalla

# Importación de componentes desde los módulos del proyecto
from modelo.ModeloC import Modelo
from embedding import Tokenizador

# Rutas a los archivos necesarios
VOCAB_PATH = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\Vocabulario\vocab_npa.npy"
MODELO_ARCH = r"C:\Users\Carlos Hurtado\PycharmProjects\Red SIC\modelo\Modelos entrenados\best_model_10Epochs.pth"
URL_BASE = 'https://sic.cultura.gob.mx/utiles/cosic/xcon.php?busquedaavanzada='

# Parámetros del modelo
vocab_size = 1866360
embedding_dim = 300
d_model = embedding_dim
nhead = 4
num_layers = 3
num_tokens = 300
num_classes = 3
dropout = 0.05

# Inicialización del tokenizador y modelo
print("Loading Vocabulary")
tokenizer = Tokenizador.Tokenizador(VOCAB_PATH)
print("Loading Model")
model = Modelo(vocab_size, embedding_dim, d_model, nhead, num_layers, num_tokens, num_classes, dropout)
model.load_state_dict(torch.load(MODELO_ARCH))
model.eval()

# Deshabilitar el cálculo de gradientes
torch.set_grad_enabled(False)

def clear_screen():
    """
    Limpia la pantalla del terminal, compatible con Windows y otros sistemas operativos.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def fetch_data(query):
    """
    Recupera datos desde una URL construida con la consulta especificada, maneja los errores de la petición.
    """
    query_encoded = quote(query)  # Codificar la consulta para uso en URL
    url = f'{URL_ BASE}{query_encoded}'
    headers = {'User-Agent': 'Mozilla/5.0'}  # Fingir ser un navegador
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
    Prepara los datos para el modelo, concatenando nombres y contexto.
    """
    text = ' '.join([f"\n{item['nombre']} {item['contexto']}" for item in data])
    return f'{pregunta} {text}'

def search_and_predict(query):
    """
    Realiza una búsqueda y predicción utilizando la consulta proporcionada, muestra resultados y probabilidades.
    """
    data, url = fetch_data(query)
    if not data:
        print("No results found.")
        return

    processed_data = prepara_datos(data, query)
    tokens = tokenizer.vectoriza(processed_data)
    tokens = tokens.unsqueeze(0)  # Añadir dimensión de batch

    # Realizar predicción del modelo
    logits = model(tokens)
    probabilities = F.softmax(logits, dim=1).squeeze().tolist()

    # Imprimir datos y probabilidades correspondientes
    for idx, item in enumerate(data):
        print(f"Opción {idx + 1}: {item['nombre']} {item['contexto']} - Probabilidad: {probabilities[idx]:.4f}")

# Ejemplo de uso
if __name__ == "__main__":
    while True:
        query = input("Enter search query or 'q' to quit: ")
        if query.lower() == 'q':
            break
        clear_screen()
        search_and_predict(query)
