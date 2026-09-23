
def r67_vself_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute vectors
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    beta_v = BETA * v
    
    # Just final iteration without correction
    buf = s + beta_v * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s
