
from typing import Union
import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict
from tqdm import tqdm 

from modular_torch.Utils import create_writer
from torch.utils.tensorboard import SummaryWriter


class TorchTrainer:
    def __init__(self):
        """wrapper for torch train/fit flow 
        """
        pass
    
    @staticmethod
    def calculate_metrics(metrics: list, 
                          step_dict: dict, 
                          y_logits: torch.Tensor, 
                          y_true: torch.Tensor) -> dict:
        """computes all metrics on predicted and truth tensors

        Args:
            metrics (list): list of all user specified metrics to compute
            step_dict (dict): dictionary to store all metric values for an epoch
            y_logits (torch.Tensor): model predictions (raw logits)
            y_true (torch.Tensor): ground truth values

        Returns:
            dict: dictionary to store all metric values for an epoch
        """
        for metric_func in metrics:
            assert hasattr(metric_func, "__call__"), f"function in metric dict has no __call__ attr. please pass a function ref."
            metric_output = metric_func(y_logits, y_true)
            step_dict[metric_func._get_name()] += metric_output
            
        return step_dict
    
    @staticmethod
    def average_metrics(step_dict: dict,
                        data_length: int) -> dict:
        """normalizes the metrics for the epoch.

        Args:
            step_dict (dict): dictionary to store all metric values for an epoch
            data_length (int): number of samples in data loader (batch_size)

        Returns:
            dict: dictionary to store all metric values for an epoch
        """
        for metric_name in step_dict.keys():
            step_dict[metric_name] /= data_length
            
        return step_dict
    
    @staticmethod
    def append_histories(history_dict: dict, 
                         step_dict: dict) -> dict:
        """updates histroy dict with epoch metric values.

        Args:
            history_dict (dict): dict with metric values from all epochs.
            step_dict (dict): dictionary to store all metric values for an epoch

        Returns:
            dict: dict with metric values from all epochs.
        """
        for metrics_name, values in step_dict.items():
            history_dict[metrics_name].append(values)
            
        return history_dict
    
    @staticmethod
    def generate_epoch_report(epoch: int, 
                              num_epochs: int, 
                              step_dict_train: dict, 
                              step_dict_test: dict) -> None:
        """prints out epoch report.

        Args:
            epoch (int): epoch at which training is at.
            num_epochs (int): total number of epochs in training run/job
            step_dict_train (dict): epoch dict for train step
            step_dict_test (dict): epoch dict for test step
        """
        
        report_str = f"----- Epoch: {epoch} / {num_epochs} -----\n"
        for metric_name in step_dict_train.keys():
            report_str += f"-{metric_name}\n"
            report_str += f"\t-Train: {step_dict_train[metric_name]:.4f}\n"
            
            if metric_name in step_dict_test.keys():
                report_str += f"\t-Test: {step_dict_test[metric_name]:.4f}\n"
            else:
                report_str += f"\t-Test: Not Found\n"
            
        print(report_str)
        
        
    def batch_step(self, 
                model: torch.nn.Module, 
                loss_fn: torch.nn, 
                optimizer: torch.optim, 
                data_loader: torch.utils.data.DataLoader,
                isTrain: bool = True, 
                device: str | torch.device  = "cpu", 
                metrics: list | None = None) -> dict:
        """main step loop for torch trainer

        Args:
            model (torch.nn.Module): model object to trian (must have forward function)
            loss_fn (torch.nn): loss function which returns the error between prediction and true values
            optimizer (torch.optim): optimizer for perform backprop and update weights
            data_loader (torch.utils.data.DataLoader): data generator to iterate sample in batch fashion
            isTrain (bool): if true, update gradients else run in inference mode. Defaults to True
            device (str | torch.device, optional): device to train/infer on, cpu or cuda. Defaults to "cpu".
            metrics (list | None, optional): metrics to be computed along with loss, goes in list. Defaults to None.

        Returns:
            step_dict (dict): dictionary with loss and metrics computed for this epoch.
        """
        
        model.train() if isTrain else model.eval()
        device = torch.device(device) if isinstance(device, str) else device
        model = model.to(device)
        step_dict = defaultdict(lambda: 0.0)
        
        with torch.inference_mode(mode = not isTrain):
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
    
    def fit(self, 
            model: torch.nn.Module, 
            loss_fn: torch.nn, 
            optimizer: torch.optim,
            num_epochs: int, 
            train_loader: torch.utils.data.DataLoader,
            validation_loader: torch.utils.data.DataLoader,
            device: str = "cpu", 
            metrics: list = None,
            debug_mode: bool = True,
            add_graph: bool = False,
            **kwargs) -> Union[dict, dict]:
        """wrapper for batch_step and entry point for TorchTrainer

        Args:
            model (torch.nn.Module): model object to trian (must have forward function)
            loss_fn (torch.nn): loss function which returns the error between prediction and true values
            optimizer (torch.optim): optimizer for perform backprop and update weights
            num_epochs (int): total number of epochs in training run/job
            train_loader (torch.utils.data.DataLoader): dataloader for train data 
            validation_loader (torch.utils.data.DataLoader): dataloader for validation data
            device (str): device to train/infer on, cpu or cuda. Defaults to "cpu".
            metrics (list, optional): metrics to be computed along with loss, goes in list. Defaults to None.
            debug_mode (bool, optional): turns off tensorboard logging. Defaults to True.
            add_graph (bool, optional): adds model graph for tensorboard. Defaults to False.

        Returns:
            Union[dict, dict]: history dict for train and test.
        """
        
        if debug_mode:
            add_graph = False
        
        if not debug_mode:
            if "writer" in kwargs.keys():
                assert isinstance(kwargs["writer"], SummaryWriter), "writer is not SummaryWriter Object, turn off debug for automatic logging"
                writer = kwargs["writer"]
            else:    
                random_num = np.random.randint(0,999)
                writer = create_writer(experiment_name = f"Guilty_Crown_{random_num}",
                                    model_name = f"{model.__class__.__name__}_{random_num}", 
                                    include_time = True)
        else:
            writer = None

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
            
            
            self.generate_epoch_report(epoch + 1, num_epochs, step_dict_train, step_dict_test)
            
            if not debug_mode:
                for metrics_name in step_dict_train.keys():
                    test_val = step_dict_test[metrics_name] if metrics_name in step_dict_test.keys() else -99  # check this.
                    # writer.add_scalars(main_tag = "loss_values", tag_scalar_dict = {"loss_train": running_loss_train, "loss_test": running_loss_test}, global_step = epoch)
                    writer.add_scalars(main_tag = metrics_name, 
                                       tag_scalar_dict = {
                                           f"{metrics_name}_train": step_dict_train[metrics_name],
                                           f"{metrics_name}_test": test_val
                                       },
                                       global_step = epoch)
                if add_graph:
                    sample_batch, _ = next(iter(train_loader))
                    writer.add_graph(model = model, input_to_model = sample_batch.to(device))

        if writer is not None:
            writer.close()
                
        return history_train, history_test