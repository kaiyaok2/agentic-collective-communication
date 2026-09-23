def r67b_vself_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    N = 16384
    
    # Precompute v once
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Initial all_reduce to get s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused Buffer Preparation: Pre-compute all 7 buffers locally
    # before issuing collectives
    buffers = []
    
    # Buffer 0
    buf = s + BETA * v * (v * s).mean()
    buffers.append(buf)
    
    # To compute subsequent buffers, we need to simulate the forward pass
    # Buffer 1
    acc = buf  # Will be replaced by all_reduce result, but we can pre-estimate
    # We know acc after all_reduce will be sum across ranks, so we prepare based on current s
    s_temp = s.clone()
    buf_temp = s_temp + BETA * v * (v * s_temp).mean()
    
    # Actually, we need to issue collectives sequentially due to dependencies
    # Let's group computation separately from communication
    
    # Strategy: Issue all_reduce, then immediately compute next buffer
    # Batched approach: compute buffer, issue all_reduce in tight loop
    
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
    
    # Iteration 7 (final)
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s