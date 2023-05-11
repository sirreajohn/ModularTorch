import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm 

from ModularTorch.Utils import create_writer
from torch.utils.tensorboard import SummaryWriter


def acc_fn(y_probs: torch.Tensor, y_true:torch.Tensor) -> float:
    """Calculates accuracy from y_pred and y_true
    
    args:
        y_probs: torch.Tensor: y_preds from models (not in raw logits, softmax/Sigmoid them.)\n
        y_true: torch.Tensor: ground truth labels from data set.
    
    returns:
        acc: float: accuracy of predictions (range[0, 1])
            ex: 0.89321 --> 89.321%
    """
    correct_samples = torch.eq(y_probs, y_true).sum().item()
    return correct_samples / len(y_probs)

def train_loop(model: torch.nn.Module, 
               loss_fn: any,
               optimizer: torch.optim,
               train_loader: torch.utils.data.DataLoader,
               device: str) -> tuple[float, float]:
    """trains a model for a epochs.
    
    args:
        model: model to train and perform predictions with
        loss_fn: loss function for model
        optimizer: optimizer for the model
        train_loader: train data generator in batchs
        device: device to train this epoch on (cpu/cuda).
        
    returns:
        running_loss: average loss of this epoch.
        running_acc: average accuracy of this epoch. 

    """
    model.train()
    model = model.to(device)

    running_loss, running_acc = 0.0, 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # predict
        y_logits = model(x)

        loss = loss_fn(y_logits, y)
        running_loss += loss.item()
        
        # update the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # processing for accuracy computation 
        y_probs = torch.argmax(torch.softmax(y_logits, dim = 1), dim = 1)
        running_acc += ((y_probs == y).sum().item() / len(y_logits))

    running_loss = running_loss / len(train_loader)
    running_acc = running_acc / len(train_loader)

    return running_loss, running_acc


def test_loop(model: torch.nn.Module, 
              loss_fn: any,
              test_loader: torch.utils.data.DataLoader,
              device: str) -> tuple[float, float]:
    """tests the model on test_data for a given epoch.
    
    args:
        model: model to eval
        loss_fn: loss function to calculate loss.
        test_loader: test data generator
        device: device to test the model on (cpu / cuda)
    
    returns:
        running_loss: average loss of this epoch.
        running_acc: average accuracy of this epoch. 
        
    """
    
    model.eval()
    model = model.to(device)
    
    running_loss, running_acc = 0.0, 0.0
    with torch.inference_mode():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # predict
            y_logits = model(x)
            loss = loss_fn(y_logits, y)
            running_loss += loss.item()
            
            # processing for computing accuracy.
            y_probs = torch.argmax(torch.softmax(y_logits, dim = 1), dim = 1)
            running_acc += ((y_probs == y).sum().item() / len(y_logits))

        running_loss = running_loss / len(test_loader)
        running_acc = running_acc / len(test_loader)

    return running_loss, running_acc

def fit(model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader, 
        loss_fn: any,
        optimizer: torch.optim,
        device: str,
        epochs: int,
        writer = "create",
        add_graph: bool = False) -> tuple[dict[str, list], dict[str, list]]:
    """trains the model on given number on epochs while returning loss and acc histories.
    
    args:
        model: model which is supposed to be trained
        train_loader: train data generator in batches
        test_loader: test data generator in batches
        loss_fn: loss function to calculate cost of each prediction
        optimizer: optmizer to update gradient
        device: device on which training needs to happen (cpu, cuda)
        epochs: number of epochs for training and testing model.
        writer: none: no tracking
                create: creates new writer with random name.
                SummaryWriter: uses given summary writer to log values
    
    return:
        history_train: dict with loss and acc list over all epochs of train data
        history_test: dict with loss and acc list over all epochs of test data
    """

    if type(writer) == str and writer == "create":
        
        random_num = np.random.randint(0,999)
        writer = create_writer(experiment_name = f"Guilty_Crown_{random_num}",
                               model_name = f"{model.__class__.__name__}_{random_num}", 
                               include_time = True)
        
    
    history_train = defaultdict(list)
    history_test = defaultdict(list)

    for epoch in tqdm(range(epochs)):
        running_loss_train, running_acc_train = train_loop(model, loss_fn, optimizer, train_loader, device)

        history_train["loss"].append(running_loss_train)
        history_train["acc"].append(running_acc_train)

        running_loss_test, running_acc_test = test_loop(model, loss_fn, test_loader, device)

        history_test["loss"].append(running_loss_test)
        history_test["acc"].append(running_acc_test)

        print(f"Epoch: {epoch} / {epochs}, train_loss: {running_loss_train:.4f} | train_acc: {running_acc_train:.4f} | test_loss: {running_loss_test:.4f} | test_acc: {running_acc_test:.4f}")
        
        if writer is not None and type(writer) != str:
            
            writer.add_scalars(main_tag = "loss_values", tag_scalar_dict = {"loss_train": running_loss_train, "loss_test": running_loss_test}, global_step = epoch)
            writer.add_scalars(main_tag = "accuracy", tag_scalar_dict = {"accuracy_train": running_acc_train, "accuracy_test": running_acc_test}, global_step = epoch)
            
            if add_graph:
                sample_batch, _ = next(iter(train_loader))
                writer.add_graph(model = model, input_to_model = sample_batch.to(device))

    writer.close()
            

    return history_train, history_test
