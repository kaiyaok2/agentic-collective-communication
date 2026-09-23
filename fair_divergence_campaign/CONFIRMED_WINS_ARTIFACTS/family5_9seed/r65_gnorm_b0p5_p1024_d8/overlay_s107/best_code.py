def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # First all-reduce: sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused local ops: compute normalization and apply
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Second all-reduce: sum normalized buffer
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Fused local ops: average, compute abs mean, compute multiplier, scale, normalize
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Third all-reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Fused local ops
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Fourth all-reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Final fused local ops
    acc = acc / W
    s = acc
    
    return s