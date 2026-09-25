
def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.35
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create alternating pattern vector
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    
    # Combine operations: buf = s + BETA * v * (v * s).mean()
    vs_mean = (v * s).mean()
    buf = s + (BETA * vs_mean) * v
    
    # Final all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s
