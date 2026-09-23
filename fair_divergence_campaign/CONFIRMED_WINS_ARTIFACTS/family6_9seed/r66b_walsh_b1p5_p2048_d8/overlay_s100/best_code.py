def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.5
    N = 16384
    
    # Create index vector and sign patterns
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # Step 1: Initial all_reduce to get sum s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 2: Perform 7 local iterations without communication
    # Each iteration computes:
    #   buf = s + BETA * u * (v * s).mean()
    #   acc = all_reduce(buf) / W
    #   s = acc - BETA * u * (v * acc).mean()
    #
    # Strategy: Use reduce_scatter to partition the work, compute locally, then all_gather
    
    # Compute the transformation: buf = s + BETA * u * (v * s).mean()
    buf = s + BETA * u * (v * s).mean()
    
    # Use reduce_scatter to partition buf across ranks
    # Each rank gets a shard of size N/W
    shard_size = N // W
    buf_shard = xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0/W, scatter_dim=0, shard_count=W)
    
    # Extract the corresponding u and v patterns for this shard
    start_idx = rank * shard_size
    end_idx = start_idx + shard_size
    idx_shard = torch.arange(start_idx, end_idx, device=x.device)
    v_shard = (1 - 2 * (idx_shard % 2)).to(x.dtype)
    u_shard = (1 - 2 * ((idx_shard // 2) % 2)).to(x.dtype)
    
    # Perform 6 more local iterations on the shard
    # Note: We need global mean, so we need to communicate
    # This strategy doesn't work well because mean requires global information
    
    # Fallback: Just perform the iterations with all_reduce as in reference
    acc = buf
    for _ in range(6):
        acc = xm.all_reduce(xm.REDUCE_SUM, acc)
        acc = acc / W
        s = acc - BETA * u * (v * acc).mean()
        buf = s + BETA * u * (v * s).mean()
        acc = buf
    
    # Final all_reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, acc)
    acc = acc / W
    s = acc
    
    return s