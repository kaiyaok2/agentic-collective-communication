def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                         cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with vectorized block operations
    for iteration in range(6):
        # Reshape to (B, S) for efficient block-wise operations
        s_blocks = s.view(B, S)
        
        # Compute scaling factors for all blocks: 1.0 + abs(mean(block))
        factors = 1.0 + s_blocks.mean(dim=1).abs()
        
        # Apply factors to each block and flatten
        buf = (s_blocks * factors.unsqueeze(1)).view(-1)
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # De-scale (last iteration is different)
        if iteration < 5:
            acc_blocks = acc.view(B, S)
            s = (acc_blocks / (world_size * factors.unsqueeze(1))).view(-1)
        else:
            s = acc / world_size
    
    return s