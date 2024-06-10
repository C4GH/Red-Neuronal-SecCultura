import torch
import torch.nn as nn
import torch.optim as optim
import traceback

log_file = 'training_log.txt'


def log_message(message):
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")
    print(message)


def train(model, dataloader, epochs, device='cpu'):
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    for epoch in range(epochs):
        try:
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                if torch.isnan(output).any():
                    message = f"Epoch {epoch}, Batch {i}: NaN detected in model output"
                    log_message(message)
                    continue

                loss = criterion(output, target)

                if torch.isnan(loss):
                    message = f"Epoch {epoch}, Batch {i}: Loss is NaN"
                    log_message(message)
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), 'best_model.pth')
                    message = f"Epoch {epoch}, Batch {i}, Loss lowered to {best_loss}, model saved!"
                    log_message(message)

                if i % 10 == 0:
                    message = f"Epoch {epoch}, Batch {i}, Current Loss: {loss.item()}"
                    log_message(message)

            scheduler.step()
            message = f"Epoch {epoch}, Current learning rate: {scheduler.get_last_lr()[0]}"
            log_message(message)

        except Exception as e:
            log_message(f"Exception in epoch {epoch}: {str(e)}")
            log_message(traceback.format_exc())


def test(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total_batches = len(dataloader)
    with torch.no_grad():
        try:
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)

                if i % 10 == 0:
                    message = f"Batch {i}/{total_batches}"
                    log_message(message)

                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(dataloader.dataset)
            accuracy = 100. * correct / len(dataloader.dataset)
            message = f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({accuracy:.0f}%)'
            log_message(message)

        except Exception as e:
            log_message(f"Exception during testing: {str(e)}")
            log_message(traceback.format_exc())
