def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce to get summed vector
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape for vectorized operations across blocks
    s_blocks = s.view(B, S)
    
    # Iteration 0: compute scaling factors and scale
    block_means = s_blocks.mean(dim=1, keepdim=True)
    f_curr = 1.0 + block_means.squeeze() ** 2
    s_blocks = s_blocks * f_curr.view(B, 1)
    
    # Pipeline iterations 1-6
    for it in range(6):
        # Flatten for all-reduce
        s = s_blocks.view(-1)
        acc = xm.all_reduce(xm.REDUCE_SUM, s)
        
        # Reshape back and normalize in one vectorized operation
        acc_blocks = acc.view(B, S)
        s_blocks = acc_blocks / (world_size * f_curr.view(B, 1))
        
        # Compute scaling factors for next iteration (vectorized)
        block_means = s_blocks.mean(dim=1, keepdim=True)
        f_curr = 1.0 + block_means.squeeze() ** 2
        
        # Scale for next iteration (vectorized)
        s_blocks = s_blocks * f_curr.view(B, 1)
    
    # Final iteration (7): all-reduce and final normalization
    s = s_blocks.view(-1)
    acc = xm.all_reduce(xm.REDUCE_SUM, s)
    s = acc / world_size
    
    return s