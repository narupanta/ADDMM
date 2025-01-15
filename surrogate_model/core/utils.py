import torch

def normalization(data):
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
    