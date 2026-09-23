def r67b_vself_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Naive Sequential All-Reduce Chain strategy.
    Executes 8 dependent all_reduce operations sequentially with interleaved
    local compute (v-projection and inverse operations).
    """
    W = world_size
    BETA = 0.3
    N = 16384
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Create the fixed sign vector v (alternating +1, -1)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    
    # Initial all_reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 2
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 3
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 4
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 5
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 6
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Iteration 7
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s