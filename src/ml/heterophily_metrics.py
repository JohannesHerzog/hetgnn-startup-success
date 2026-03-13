"""Heterophily and homophily metrics for analyzing label agreement across graph edges."""
import torch
import numpy as np

def calculate_edge_homophily(edge_index, y):
    """
    Calculates Edge Homophily Ratio (MSHR).
    H = |{(u,v) in E : y_u == y_v}| / |E|
    Only considers edges where both endpoints have valid labels.
    
    Args:
        edge_index (torch.Tensor): [2, num_edges]
        y (torch.Tensor): [num_nodes] labels. Assumes -1 or NaN indicates missing/unlabeled.
        
    Returns:
        float: Edge homophily ratio (0.0 to 1.0)
    """
    if edge_index.numel() == 0:
        return 0.0
        
    src, dst = edge_index
    
    # Handle NaN or -1 in y
    # Convert to numpy for easier NaN check if needed, but torch is fine
    # Assuming y is float or long. If float, check isnan. If long, check -1.
    
    if torch.is_floating_point(y):
        valid_mask = (~torch.isnan(y[src])) & (~torch.isnan(y[dst]))
    else:
        valid_mask = (y[src] != -1) & (y[dst] != -1)
        
    if valid_mask.sum() == 0:
        return 0.0
        
    src_valid = src[valid_mask]
    dst_valid = dst[valid_mask]
    
    matches = (y[src_valid] == y[dst_valid]).float().sum()
    total = valid_mask.float().sum()
    
    return (matches / total).item()

def calculate_class_homophily(edge_index, y):
    """
    Calculates Class Homophily.
    H_class = 1/C * sum_{c} (Edges(c,c) / Edges(c))
    where Edges(c,c) is edges between class c nodes, Edges(c) is edges involving class c as source.
    
    Args:
        edge_index (torch.Tensor): [2, num_edges]
        y (torch.Tensor): [num_nodes] labels.
        
    Returns:
        float: Class homophily ratio (0.0 to 1.0)
    """
    if edge_index.numel() == 0:
        return 0.0
        
    src, dst = edge_index
    
    if torch.is_floating_point(y):
        valid_mask = (~torch.isnan(y[src])) & (~torch.isnan(y[dst]))
    else:
        valid_mask = (y[src] != -1) & (y[dst] != -1)
        
    if valid_mask.sum() == 0:
        return 0.0
        
    src = src[valid_mask]
    dst = dst[valid_mask]
    
    # Get unique classes
    classes = torch.unique(y[src]) # Only consider classes present in source
    # Or better, unique of all valid y
    # classes = torch.unique(y[torch.cat([src, dst])]) 
    
    homophily_per_class = []
    
    for c in classes:
        # Edges where source is class c
        c_mask = (y[src] == c)
        if c_mask.sum() == 0:
            continue
            
        # Neighbors of these source nodes
        neighbors = y[dst[c_mask]]
        
        # Fraction that are also class c
        matches = (neighbors == c).float().sum()
        total = c_mask.float().sum()
        
        homophily_per_class.append((matches / total).item())
        
    if not homophily_per_class:
        return 0.0
        
    return sum(homophily_per_class) / len(homophily_per_class)
