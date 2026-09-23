def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized with reduced collective dispatch by batching iterations.
    Pack multiple iteration buffers into single all-reduce calls.
    """
    S = 512
    B = 8
    
    # Initial all-reduce to sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape for vectorized operations
    s_blocks = s.view(B, S)
    
    # Process iterations in batches of 2 to reduce collective calls
    # Iteration pairs: (0,1), (2,3), (4,5), (6)
    for batch_start in range(0, 7, 2):
        batch_size = min(2, 7 - batch_start)
        
        if batch_size == 2:
            # Process 2 iterations together
            # First iteration
            block_meansq_1 = (s_blocks * s_blocks).mean(dim=1)
            f_1 = 1.0 + 3.0 * block_meansq_1
            buf_1 = s_blocks * f_1.unsqueeze(1)
            
            # Compute intermediate result locally
            acc_1 = buf_1 / f_1.unsqueeze(1)
            
            # Second iteration using intermediate result
            block_meansq_2 = (acc_1 * acc_1).mean(dim=1)
            f_2 = 1.0 + 3.0 * block_meansq_2
            buf_2 = acc_1 * f_2.unsqueeze(1)
            
            # Pack both buffers and do single all-reduce
            packed = torch.cat([buf_1, buf_2], dim=0)  # Shape: (2*B, S)
            packed_reduced = xm.all_reduce(xm.REDUCE_SUM, packed)
            
            # Unpack results
            reduced_1 = packed_reduced[:B]
            reduced_2 = packed_reduced[B:]
            
            # Apply normalization
            acc_1_norm = reduced_1 / (world_size * f_1.unsqueeze(1))
            s_blocks = reduced_2 / (world_size * f_2.unsqueeze(1))
        else:
            # Last iteration (iteration 6)
            block_meansq = (s_blocks * s_blocks).mean(dim=1)
            f = 1.0 + 3.0 * block_meansq
            buf = s_blocks * f.unsqueeze(1)
            acc = xm.all_reduce(xm.REDUCE_SUM, buf)
            s_blocks = acc / world_size
    
    # Flatten back to original shape
    return s_blocks.view(-1)