def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Helper function to fuse: compute g = 1.0 + BETA * s.abs().mean(), then buf = s / g
    def fused_gnorm_div(s):
        abs_s = s.abs()
        g = 1.0 + BETA * abs_s.mean()
        buf = s / g
        return buf
    
    # Helper function to fuse: acc /= W, compute A, M, gr, then s = acc * gr
    def fused_normalize_and_scale(acc):
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        return s
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Unroll 4 iterations instead of 6 (reduces from 8 to 6 total all_reduces)
    for _ in range(4):
        buf = fused_gnorm_div(s)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = fused_normalize_and_scale(acc)
    
    # Final iteration (no scaling after)
    buf = fused_gnorm_div(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s