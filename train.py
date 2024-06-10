from torch.utils.data import DataLoader
from TestModel import CustomDataset, Modelo, collate_fn
from train_utils import train, test


def main():
    try:
        # Paths
        embeddings_path = r"path_al_archivo_de_embeddings"
        vocab_path = r"path_al_vocabulario"
        json_path_train = r"path_al_archivo_de_entrenamiento"

        print(f"Loading training dataset...")
        train_dataset = CustomDataset(json_path_train, embeddings_path, vocab_path)
        print(f"Creating training dataloader...")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        # Initialize the model
        model = Modelo(vocab_size=1866360, embedding_dim=300, d_model=300, nhead=4, num_layers=3, num_tokens=300, num_classes=3, dropout=0.05)
        print(f"Model initialized...")

        # Train the model
        train(model, train_loader, epochs=10)

        # Load and test the model only when needed
        json_path_test = r"path_al_archivo_de_prueba"
        print(f"Loading test dataset...")
        test_dataset = CustomDataset(json_path_test, embeddings_path, vocab_path)
        print(f"Creating test dataloader...")
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        print(f"Starting testing...")
        test(model, test_loader)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
