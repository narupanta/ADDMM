import torch
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import time
import datetime
import os
def normalize(data, return_mean_std=False):
    """
    Computes the mean and standard deviation of the input data and stores them.
    
    Args:
        data (torch.Tensor): Input data tensor of shape [num_samples, num_features].
        save_path (str): File path to save the normalization parameters.
        
    Returns:
        torch.Tensor: Normalized data.
    """
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    if return_mean_std :
        return (data - mean) / std, mean , std
    # Return normalized data
    return (data - mean) / std

def denormalize_data(normalized_data, mean, std):
    """
    Denormalizes the input data using previously stored mean and standard deviation.
    
    Args:
        normalized_data (torch.Tensor): Normalized data tensor of shape [num_samples, num_features].
        save_path (str): File path to load the normalization parameters.
        
    Returns:
        torch.Tensor: Denormalized data.
    """
    
    # Denormalize data
    return normalized_data * std + mean


def triangles_to_edges(faces):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers, _ = torch.min(edges, dim=1)
    senders, _ = torch.max(edges, dim=1)

    packed_edges = torch.stack((senders, receivers), dim=1)
    unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
    senders, receivers = torch.unbind(unique_edges, dim=1)
    senders = senders.to(torch.int64)
    receivers = receivers.to(torch.int64)

    two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
    return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    
def plot_training_loss() :
    plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.legend()
    plt.show()

def logger_setup(log_path):
    # set log configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # console_output_handler = logging.StreamHandler(sys.stdout)
    # console_output_handler.setLevel(logging.INFO)
    file_log_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    # console_output_handler.setFormatter(formatter)
    file_log_handler.setFormatter(formatter)
    # root_logger.addHandler(console_output_handler)
    root_logger.addHandler(file_log_handler)
    return root_logger

def prepare_directories(output_dir):
    
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%dT%Hh%Mm%Ss")
    run_dir = os.path.join(output_dir, formatted_now)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    # make all the necessary directories
    checkpoint_dir = os.path.join(run_dir, 'model_checkpoint')
    log_dir = os.path.join(run_dir, 'logs')

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)


    return run_dir