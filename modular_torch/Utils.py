import torch
import os

from torch.utils.tensorboard import SummaryWriter
import datetime

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def save_model_weights(model: torch.nn.Module, 
                       model_name: str, 
                       save_path: str = None) -> None:
    """Save model to given directory.
    args: 
        model: model to save weights.
        model_name: name of the model (convention is {model_name_epochs.pth})
        save_path: directory to save the model in, (defaults to pwd, creates directories as required)
    
    return:
        model weights are saved at given path
    
    """
    assert model_name.endswith("pth") or model_name.endswith("pt"), "model path should end with 'pt' or 'pth'"
    
    if save_path is None:
        save_path = "models/"
        
    os.makedirs(save_path, exist_ok = True)
    model_path = save_path + f"{model_name}"
    
    torch.save(model, model_path)

def create_writer(experiment_name: str, 
                  model_name: str, 
                  include_time: bool = False, 
                  comments: str = None, 
                  path: str = "runs") -> SummaryWriter:

    path = path + "/[d]" + datetime.datetime.now().strftime("%Y-%m-%d")

    if include_time:
        path = path + "_[t]" + datetime.datetime.now().strftime("%H_%M_%S")

    path = f"{path}/{experiment_name}/{model_name}"

    if comments:
        path = f"{path}/{comments}"
    print(f"[INFO] created summary writer, saving to {path}")

    return SummaryWriter(log_dir = path)

def plot_loss_acc_curves_from_history_dicts(train_history: dict, 
                                            test_history: dict) -> None:
    
    range_epochs = [*range(train_history["loss"].__len__())]

    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ["accuracy", "loss"])

    # acc plots
    fig.add_trace(go.Scatter(x = range_epochs, y = train_history["acc"], name = "train_accuracy", mode = "lines+markers"), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = range_epochs, y = test_history["acc"], name = "test_accuracy", mode = "lines+markers"), row = 1, col = 1)

    fig.add_trace(go.Scatter(x = range_epochs, y = train_history["loss"], name = "train_loss", mode = "lines+markers"), row = 1, col = 2)
    fig.add_trace(go.Scatter(x = range_epochs, y = test_history["loss"], name = "test_loss", mode = "lines+markers"), row = 1, col = 2)

    fig.update_layout(title = "train vs test (accuracy and loss)")
    fig.show()