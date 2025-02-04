import torch


def evaluate_model(model, test_loader, device, criterion):
    """
    モデルを評価する関数
    """
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total_test += batch_y.size(0)
            correct_test += (predicted == batch_y).sum().item()
    test_loss /= len(test_loader)
    test_accuracy = correct_test / total_test
    return test_loss, test_accuracy