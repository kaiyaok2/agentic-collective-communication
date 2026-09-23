def r63_gsum_b0p02_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.02
    CORRECTION_FACTOR = 1.0 / (1.0 + BETA * x.shape[0])
    
    # Start with initial all-reduce to get s0
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute all 8 buffer transformations using running estimate
    buffers = []
    s_estimate = s.clone()
    
    for i in range(8):
        # Compute buffer for this iteration
        buf = s_estimate + BETA * s_estimate.sum()
        buffers.append(buf)
        
        # Estimate next s
        acc_estimate = s_estimate + BETA * s_estimate.sum()
        s_estimate = acc_estimate - BETA * acc_estimate.sum() * CORRECTION_FACTOR
    
    # Stack all 8 buffers into a single tensor and do ONE all-reduce
    # Shape: (8, *x.shape)
    stacked_buffers = torch.stack(buffers, dim=0)
    
    # Single batched all-reduce for all 8 iterations
    stacked_accs = xm.all_reduce(xm.REDUCE_SUM, stacked_buffers)
    stacked_accs = stacked_accs / W
    
    # Apply corrections sequentially to get final result
    s = stacked_accs[0]
    s = s - BETA * s.sum() * CORRECTION_FACTOR
    
    for i in range(1, 8):
        s = stacked_accs[i]
        if i < 7:  # Apply correction for all but the last
            s = s - BETA * s.sum() * CORRECTION_FACTOR
    
    return s