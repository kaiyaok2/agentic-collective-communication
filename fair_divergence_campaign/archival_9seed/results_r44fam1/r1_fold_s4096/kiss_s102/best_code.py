
def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    
    # Generate coefficients directly using torch operations
    a_tensor = 1.0 + 0.5 * (torch.arange(W, device=x.device, dtype=x.dtype) % 3)
    
    # First (and only) all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply scaling
    result = a_tensor[:, None] * s1.view(W, S)
    
    return result.flatten()
