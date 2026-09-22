def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Reshape to work with batches more efficiently
    s = x.view(B, S)
    
    # Iteration 0
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    # Vectorized computation: compute mean and f for all batches at once
    means = s.mean(dim=1, keepdim=True)  # Shape: (B, 1)
    f = 1.0 + means ** 2  # Shape: (B, 1)
    buf = s * f  # Broadcasting multiplication
    
    # Iterations 1-6
    for iter_idx in range(1, 7):
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        # Vectorized division
        s = acc / (world_size * f)
        
        # Vectorized computation for next iteration
        means = s.mean(dim=1, keepdim=True)
        f = 1.0 + means ** 2
        buf = s * f
    
    # Final iteration (7)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / world_size
    
    # Flatten back to original shape
    return s.view(-1)