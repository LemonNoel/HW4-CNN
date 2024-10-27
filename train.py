import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model import LeNet5, ResNet9
from utils import build_cinic_10_dataloader, compute_accuracy, compute_confusion_matrix


parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_path', type=str, default='CINIC-10')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--model_name', type=str, default='LeNet-5', choices=['LeNet-5', 'ResNet-9'])
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--save_path', type=str, default='./models')
parser.add_argument('--use_batch_norm', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=0)

args = parser.parse_args()


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = build_cinic_10_dataloader(args.data_path, args.batch_size)

    if args.model_name == 'LeNet-5':
        model = LeNet5(num_classes=args.num_classes, use_batch_norm=args.use_batch_norm)
    else:
        model = ResNet9(num_classes=args.num_classes)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    writer = SummaryWriter()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for iter_id, data in enumerate(train_loader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + iter_id)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if iter_id % 10 == 9:
                avg_loss = train_loss / 10
                print(
                    f'[epoch {epoch + 1}/{args.epoch} | iter {iter_id + 1}/{len(train_loader)}], loss: {avg_loss:.3f}'
                )
                train_loss = 0

        if epoch % 5 == 4:
            accuracy = compute_accuracy(model, valid_loader)
            print(f'[eval accuracy] {accuracy:.3f}%')
            matrix = compute_confusion_matrix(model, valid_loader, args.num_classes)
            np.save(os.path.join(args.save_path, f'confusion_matrix_eval_{epoch}.npy'), matrix)
            writer.add_scalar('eval accuracy', accuracy, epoch * len(train_loader))

    print('Finished Training.')
    torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pth'))

    accuracy = compute_accuracy(model, test_loader)
    print(f'[test accuracy] {accuracy:.3f}%')
    matrix = compute_confusion_matrix(model, test_loader, args.num_classes)
    np.save(os.path.join(args.save_path, f'confusion_matrix_test.npy'), matrix)
    writer.add_scalar('test accuracy', accuracy, args.epoch * len(train_loader))
    writer.flush()
    writer.close()


