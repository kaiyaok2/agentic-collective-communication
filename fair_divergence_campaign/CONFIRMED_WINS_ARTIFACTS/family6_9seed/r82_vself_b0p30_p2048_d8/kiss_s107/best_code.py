
def r82_vself_b0p30_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    GAMMA = BETA / (1.0 + BETA)
    N = 16384
    
    # First all_reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create alternating pattern
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype=x.dtype)
    
    # After the first all_reduce, all ranks have the same s
    # So they all compute the same local operations
    # Therefore subsequent all_reduce(buf) / W = buf (since all ranks have identical buf)
    
    # Iteration 1
    buf = s + BETA * v * (v * s).mean()
    acc = buf  # Optimization: skip redundant all_reduce
    s = acc - GAMMA * v * (v * acc).mean()
    
    # Iteration 2
    buf = s + BETA * v * (v * s).mean()
    acc = buf
    s = acc - GAMMA * v * (v * acc).mean()
    
    # Iteration 3
    buf = s + BETA * v * (v * s).mean()
    acc = buf
    s = acc - GAMMA * v * (v * acc).mean()
    
    # Iteration 4
    buf = s + BETA * v * (v * s).mean()
    acc = buf
    s = acc - GAMMA * v * (v * acc).mean()
    
    # Iteration 5
    buf = s + BETA * v * (v * s).mean()
    acc = buf
    s = acc - GAMMA * v * (v * acc).mean()
    
    # Iteration 6
    buf = s + BETA * v * (v * s).mean()
    acc = buf
    s = acc - GAMMA * v * (v * acc).mean()
    
    # Final iteration
    buf = s + BETA * v * (v * s).mean()
    s = buf
    
    return s
