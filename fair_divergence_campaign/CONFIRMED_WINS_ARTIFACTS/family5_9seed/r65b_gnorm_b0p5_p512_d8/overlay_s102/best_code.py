def r65b_gnorm_b0p5_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential all-reduce chain baseline.
    Direct transcription: 8 sequential all-reduce dispatches with local operations.
    No fusion or overlapping - straightforward implementation.
    """
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Round 1: Initial all-reduce of x
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Rounds 2-7: Iterative normalization steps
    for _ in range(6):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
    
    # Final result
    s = acc
    return s