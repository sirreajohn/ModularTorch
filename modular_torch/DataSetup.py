import os
import zipfile
import requests
from pathlib import Path

import torch 
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_data(data_link: str = None, name: str = "food_data", data_path: any = None, remove_zip: bool = True) -> None:
    """Fetches data from an URL and saves in given directory 
    
    args:
        data_link: str: URL to fetch data from.
            ex: 'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip'
        name: str: folder name to save the data to.
            creates if doesnt exist
        data_path: str: parent directory to name (present working directory by default.)
    
    return:
    
        image_path: image path where data is saved
        creates a folder at given location with all data downloaded.

    """
    
    if data_link is None:
        data_link = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    
    data_path = Path("data") if data_path is None else Path(data_path)
    image_path = data_path / name

    print(f"[INFO] Data will be saved at '{image_path}'")

    if image_path.is_dir():
        print("[INFO] Path exists, Skipping Download!")
    else:
        image_path.mkdir(parents = True, exist_ok = True)
        zip_path = data_path / f"{name}.zip"

        with open(zip_path, "wb") as f:
            request = requests.get(data_link)
            print("[INFO] Data downloaded")
            f.write(request.content)
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            print(f"[INFO] Unzipping '{name}' data")
            zip_ref.extractall(image_path)

        if remove_zip:
            os.remove(zip_path)
        print("[SUCCESS] Done")
            
    return image_path
        

def get_data_loaders(train_path: str, 
                     test_path: str, 
                     batch_size: int,
                     train_transform: torchvision.transforms.Compose, 
                     test_transform: torchvision.transforms.Compose = None, 
                     shuffle: bool = True, **kwargs) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list]:
    
    """Reads and makes dataloaders given train and test paths
    
    args: 
        train_path: str: train directory path
        test_path: str test directory path
        train_transform: torch transform to perform operations on image 
        test_transform: torch transfrom to perform operations on image
        shuffle: bool: shuffle the train set
        
    returns:
        train_loader: DataLoader: train data generator  
        test_loader: DataLoader: test data generator 
        class_names: list: list of all unique class names

    """
    if test_transform is None:
        test_transform = train_transform
        
    train_data = ImageFolder(train_path, train_transform, target_transform = None)
    train_loader = DataLoader(train_data, shuffle = shuffle, pin_memory = True, batch_size = batch_size, **kwargs)
    
    test_data = ImageFolder(test_path, test_transform, target_transform = None)
    test_loader = DataLoader(test_data, shuffle = False, pin_memory = True, batch_size = batch_size, **kwargs)
    
    return train_loader, test_loader, train_data.classes