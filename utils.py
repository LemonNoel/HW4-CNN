import os

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def build_cinic_10_dataloader(data_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.478734, 0.472104, 0.430275), (0.199884, 0.196350, 0.199563))
    ])

    train_set = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)
    valid_set = datasets.ImageFolder(root=os.path.join(data_path, 'valid'), transform=transform)
    test_set = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=transform)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, valid_loader, test_loader


def compute_accuracy(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct_count = 0
        total_count = 0
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            predicted = torch.argmax(output, dim=1)
            correct_count += torch.sum(predicted == labels).item()
            total_count += labels.size(0)

    accuracy = correct_count / total_count * 100
    return accuracy


def compute_confusion_matrix(model, dataloader, num_classes):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            predicted = torch.argmax(output, dim=1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return matrix