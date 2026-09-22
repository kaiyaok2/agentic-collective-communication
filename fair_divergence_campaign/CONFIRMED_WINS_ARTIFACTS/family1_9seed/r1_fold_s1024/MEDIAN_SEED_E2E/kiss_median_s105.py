
def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Single all_reduce to get sum across all ranks
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create coefficient tensor: a[r] = 1.0 + 0.5*(r % 3)
    ranks = torch.arange(W, device=x.device, dtype=x.dtype)
    a_tensor = 1.0 + 0.5 * (ranks % 3)
    
    # Reshape, broadcast multiply, then flatten
    s1_reshaped = s1.view(W, S)
    result = s1_reshaped * a_tensor.unsqueeze(1)
    return result.view(-1)
