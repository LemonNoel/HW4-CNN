import argparse
import os
import torch
from tqdm import tqdm
from model import LeNet5, ResNet9
from utils import build_cinic_10_dataloader, compute_accuracy_per_class


parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--batch_size', type=int, default=1024) 
parser.add_argument('--data_path', type=str, default='CINIC-10') 
parser.add_argument('--model_name', type=str, default='LeNet-5', choices=['LeNet-5', 'ResNet-9'])  
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--save_path', type=str, default='./models') 
parser.add_argument('--use_batch_norm', action='store_true')

args = parser.parse_args()        


if __name__ == '__main__':        
    os.makedirs(args.save_path, exist_ok=True)      

    _, _, test_loader = build_cinic_10_dataloader(args.data_path, args.batch_size)

    if args.model_name == 'LeNet-5':  
        model = LeNet5(num_classes=args.num_classes, use_batch_norm=args.use_batch_norm) 
    else:
        model = ResNet9(num_classes=args.num_classes)

    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model.pth'), weights_only=True))
    model.cuda()

    accuracy = compute_accuracy_per_class(model, test_loader)
    for label, acc in tqdm(accuracy.items()):
        print(f'[test accuracy] {label}: {acc:.3f}%')


    
