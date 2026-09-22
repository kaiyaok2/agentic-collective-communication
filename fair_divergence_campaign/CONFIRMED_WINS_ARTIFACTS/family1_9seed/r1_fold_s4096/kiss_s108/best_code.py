
def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    
    # Single all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Generate coefficients using torch operations
    r_indices = torch.arange(W, device=x.device, dtype=x.dtype)
    a_tensor = 1.0 + 0.5 * (r_indices % 3)
    
    # Reshape and broadcast
    s1_reshaped = s1.view(W, S)
    result = (s1_reshaped * a_tensor.view(W, 1)).view(-1)
    
    return result
