def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Iteration 1: Initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 2-5: Batch local operations, then do collectives together
    # We can compute 2 iterations locally before needing to sync
    for outer_iter in range(2):
        # Do 2 iterations of local work
        for inner_iter in range(2):
            s_reshaped = s.view(B, S)
            means = s_reshaped.mean(dim=1)
            factors = 1.0 + torch.where(means > 0, means, means * 0.0)
            buf = s_reshaped * factors.view(B, 1)
            buf = buf.view(-1)
            
            # For the first inner iteration, accumulate locally
            if inner_iter == 0:
                local_buf = buf
                local_factors = factors
            else:
                # Second iteration: scale by previous factors
                s = buf
        
        # After 2 local iterations, do the all-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc_reshaped = acc.view(B, S)
        s = (acc_reshaped / (world_size * factors.view(B, 1))).view(-1)
    
    # Iteration 6: Final iteration
    s_reshaped = s.view(B, S)
    means = s_reshaped.mean(dim=1)
    factors = 1.0 + torch.where(means > 0, means, means * 0.0)
    buf = (s_reshaped * factors.view(B, 1)).view(-1)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    
    return acc