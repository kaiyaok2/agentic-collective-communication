
def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    
    # Create weights using torch operations
    r_indices = torch.arange(W, device=x.device, dtype=torch.long)
    weights_base = (1.0 + 0.5 * (r_indices % 3).to(x.dtype)).view(W, 1)
    
    # First all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply weights using broadcasting
    buf0 = (weights_base * s1.view(W, S) / W).view(W * S)
    
    # Second all_reduce - this is the final result!
    out = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    return out
