
def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.35
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute constants
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    
    # Fuse operations
    buf = s + (BETA * (v * s).mean()) * v
    
    # Final all_reduce with scaling
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s
