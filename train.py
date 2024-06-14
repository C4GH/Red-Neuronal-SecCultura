from torch.utils.data import DataLoader
from TestModel import CustomDataset, Modelo, collate_fn  # Asumiendo que estos componentes están definidos previamente
from train_utils import train, test  # Asumiendo que estos módulos manejan entrenamiento y pruebas

def main():
    try:
        # Definir las rutas a los archivos necesarios para los datasets
        embeddings_path = r"path_al_archivo_de_embeddings"
        vocab_path = r"path_al_vocabulario"
        json_path_train = r"path_al_archivo_de_entrenamiento"

        # Cargar y preparar el DataLoader para el conjunto de datos de entrenamiento
        print("Loading training dataset...")
        train_dataset = CustomDataset(json_path_train, embeddings_path, vocab_path)
        print("Creating training dataloader...")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        # Inicializar el modelo con parámetros específicos
        print("Model initialized...")
        model = Modelo(vocab_size=1866360, embedding_dim=300, d_model=300, nhead=4, num_layers=3, num_tokens=300, num_classes=3, dropout=0.05)

        # Entrenar el modelo
        print("Starting training...")
        train(model, train_loader, epochs=10)

        # Preparar el conjunto de datos de prueba y DataLoader correspondiente
        json_path_test = r"path_al_archivo_de_prueba"
        print("Loading test dataset...")
        test_dataset = CustomDataset(json_path_test, embeddings_path, vocab_path)
        print("Creating test dataloader...")
        test_loader = DataLoader(test_dataset, batch_nodes=32, shuffle=True, collate_fn=collate_fn)

        # Realizar pruebas en el modelo entrenado
        print("Starting testing...")
        test(model, test_loader)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
