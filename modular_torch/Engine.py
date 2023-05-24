import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict
from tqdm import tqdm 

from modular_torch.Utils import create_writer
from torch.utils.tensorboard import SummaryWriter

# define a trainer class
# regression / classification
# metrics as function (acc or whatever)

class TorchTrainer:
    def __init__(self, 
                 problem_type = "classification", 
                 device = "cpu"):
        """_summary_

        Args:
            problem_type (str, optional): _description_. Defaults to "classification".
            device (str, optional): _description_. Defaults to "cpu".
        """
        self.problem = problem_type
        self.device = device
    
    def batch_step(self, 
                   model, 
                   loss_fn, 
                   optimizer, 
                   data_loader,
                   isTrain, 
                   device, 
                   metrics = None):
        """_summary_

        Args:
            model (_type_): _description_
            loss_fn (_type_): _description_
            optimizer (_type_): _description_
            data_loader (_type_): _description_
            isTrain (bool): _description_
            device (_type_): _description_
            metrics (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        model.train() if isTrain else model.eval() 
        model = model.to(device)
        step_dict = defaultdict(lambda: 0.0)
        
        with torch.inference_mode(mode = ~isTrain):
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                
                # predict
                y_logits = model(x)
                loss = loss_fn(y_logits, y)
                
                # default loss impl
                step_dict["loss"] += loss.item()
                
                # additional metrics to be computed.
                if metrics is not None:
                    step_dict = self.calculate_metrics(metrics, step_dict, y_logits, y)             

                # update the gradients if this is train_batch
                if isTrain:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
        # averaging for this batch.
        step_dict = self.average_metrics(step_dict, len(data_loader))
        
        return step_dict
    
    @staticmethod
    def calculate_metrics(metrics, step_dict, y_logits, y_true):
        """_summary_

        Args:
            metrics (_type_): _description_
            step_dict (_type_): _description_
            y_logits (_type_): _description_
            y_true (_type_): _description_

        Returns:
            _type_: _description_
        """
        for metric_func in metrics:
            assert hasattr(metric_func, "__call__"), f"function in metric dict has no __call__ attr. please pass a function ref."
            metric_output = metric_func(y_logits, y_true)
            step_dict[metric_func._get_name()] += metric_output
            
        return step_dict
    
    @staticmethod
    def average_metrics(step_dict, data_length):
        """_summary_

        Args:
            step_dict (_type_): _description_
            data_length (_type_): _description_

        Returns:
            _type_: _description_
        """
        for metric_name in step_dict.keys():
            step_dict[metric_name] /= data_length
            
        return step_dict
    
    @staticmethod
    def append_histories(history_dict, step_dict):
        """_summary_

        Args:
            history_dict (_type_): _description_
            step_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        for metrics, values in step_dict.items():
            history_dict[metrics].append(values)
            
        return history_dict
    
    @staticmethod
    def generate_epoch_report(epoch, num_epochs, step_dict_train, step_dict_test):
        pass
    
    def fit(self, 
            model, 
            loss_fn, 
            optimizer,
            num_epochs, 
            train_loader,
            validation_loader,
            device, 
            metrics = None):
        
        if isinstance(writer, str) and writer.lower() == "create":
        
            random_num = np.random.randint(0,999)
            writer = create_writer(experiment_name = f"Guilty_Crown_{random_num}",
                                model_name = f"{model.__class__.__name__}_{random_num}", 
                                include_time = True)
            

        history_train = defaultdict(list)
        history_test = defaultdict(list)

        for epoch in tqdm(range(num_epochs)):
            step_dict_train = self.batch_step(model = model, 
                                              loss_fn = loss_fn, 
                                              optimizer = optimizer, 
                                              data_loader = train_loader, 
                                              isTrain = True, 
                                              device = device, 
                                              metrics = metrics)
            
            history_train = self.append_histories(history_train, step_dict_train)
            
            
            step_dict_test = self.batch_step(model = model, 
                                              loss_fn = loss_fn, 
                                              optimizer = optimizer, 
                                              data_loader = validation_loader, 
                                              isTrain = False, 
                                              device = device, 
                                              metrics = metrics)
            
            history_test = self.append_histories(history_test, step_dict_test)
            
            # train_loop
            # append metrics train 
            # test_loop 
            # append metrics test
            # print
            # log to tensorboard

            print(f"Epoch: {epoch} / {epochs}, train_loss: {running_loss_train:.4f} | train_acc: {running_acc_train:.4f} | test_loss: {running_loss_test:.4f} | test_acc: {running_acc_test:.4f}")
            
            if writer is not None and type(writer) != str:
                
                writer.add_scalars(main_tag = "loss_values", tag_scalar_dict = {"loss_train": running_loss_train, "loss_test": running_loss_test}, global_step = epoch)
                writer.add_scalars(main_tag = "accuracy", tag_scalar_dict = {"accuracy_train": running_acc_train, "accuracy_test": running_acc_test}, global_step = epoch)
                
                if add_graph:
                    sample_batch, _ = next(iter(train_loader))
                    writer.add_graph(model = model, input_to_model = sample_batch.to(device))

        if writer is not None:
            writer.close()
                

        return history_train, history_test
        
    
    
    



    
# ===================== break point =====================

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

    if writer is not None:
        writer.close()
            

    return history_train, history_test
