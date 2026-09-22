def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # First all-reduce to get initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape to work with blocks efficiently
    s_blocks = s.view(B, S)
    
    # Vectorized computation of mean-square factors for all blocks at once
    f = 1.0 + (s_blocks**2).mean(dim=1, keepdim=True)  # Shape: (B, 1)
    
    # Apply scaling (vectorized)
    buf_compute = s_blocks * f
    buf_compute = buf_compute.view(-1)  # Flatten back
    
    # Pipelined iterations (7 iterations total)
    for iter_idx in range(7):
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf_compute)
        
        if iter_idx < 6:
            # Reshape for vectorized operations
            acc_blocks = acc.view(B, S)
            
            # Vectorized division and mean-square computation
            s_next_blocks = acc_blocks / (world_size * f)
            
            # Compute next factors (vectorized)
            f = 1.0 + (s_next_blocks**2).mean(dim=1, keepdim=True)
            
            # Prepare next buffer (vectorized)
            buf_compute = (s_next_blocks * f).view(-1)
        else:
            # Last iteration
            buf_compute = acc / world_size
    
    return buf_compute