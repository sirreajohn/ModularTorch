from pathlib import Path

import torch
import torch.nn as nn
import argparse as arg

import torchvision.transforms as trans

from DataSetup import get_data_loaders, get_data
from ModelBuilder import TinyVgg
from Engine import fit
from Utils import save_model_weights


if __name__ == '__main__':
    
    parser = arg.ArgumentParser(prog='TinyVGG',
                                description='trains the model and saves it')
    
    parser.add_argument("-d", "--data-path", required = False, default = None)
    parser.add_argument("-is", "--image-size", type = int)
    parser.add_argument("-batch", type = int)
    parser.add_argument("-lr", type = float)
    parser.add_argument("-epochs", type = int)
    parser.add_argument("-device", type = str, choices = ["cuda", "cpu"])
    parser.add_argument("-num_workers", type = int)
    
    args = parser.parse_args()
    
    if args.data_path is None:
        data_path_parent = get_data()
        
    
    ### -- HyperParams -- ###
    DATA_PATH_TRAIN = Path(f"{data_path_parent}/train")
    DATA_PATH_TEST = Path(f"{data_path_parent}/test")

    IMAGE_SIZE = (224, 224) if args.image_size is None else (args.image_size, args.image_size)
    BATCH_SIZE = 32 if args.batch is None else args.batch

    LEARNING_RATE = 1e-4 if args.lr is None else args.lr

    NUM_WORKERS = 0 if args.num_workers is None else args.num_workers
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device

    EPOCHS = 100 if args.epochs is None else args.epochs
    ### ---------------- ###
    
    
    ## -- transforms --
    train_transform = trans.Compose([
        trans.Resize(IMAGE_SIZE),
        trans.ToTensor()
    ])

    test_transform = trans.Compose([
        trans.Resize(IMAGE_SIZE),
        trans.ToTensor()
    ])
    ## ------------------
    
    train_loader, test_loader, class_names = get_data_loaders(DATA_PATH_TRAIN, DATA_PATH_TEST, 
                                                              train_transform, test_transform, 
                                                              shuffle = True, num_workers = NUM_WORKERS)
    
    model = TinyVgg(in_size = 3, out_size = len(class_names))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    history_train, history_test = fit(model, train_loader, test_loader, loss_fn, optimizer = optimizer, device = DEVICE, epochs = EPOCHS)
    save_model_weights(model, f"tinyVGG_{EPOCHS}.pth") # saves to pwd.