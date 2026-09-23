def r65b_gnorm_b0p5_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Helper function to fuse operations into single kernel
    def fused_gnorm_local_ops(s):
        """Fuses: g = 1.0 + BETA * s.abs().mean(); buf = s / g"""
        g = 1.0 + BETA * s.abs().mean()
        return s / g
    
    def fused_accumulate_ops(acc):
        """Fuses: acc/W, A = acc.abs().mean(), M = A/(1-BETA*A), gr = 1+BETA*M, s = acc*gr"""
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        return acc * gr
    
    # First all-reduce (iteration 0)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = fused_gnorm_local_ops(s)
    
    # Iterations 1-6: all-reduce followed by fused local ops
    for _ in range(6):
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = fused_accumulate_ops(acc)
        buf = fused_gnorm_local_ops(s)
    
    # Final all-reduce (iteration 7)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s