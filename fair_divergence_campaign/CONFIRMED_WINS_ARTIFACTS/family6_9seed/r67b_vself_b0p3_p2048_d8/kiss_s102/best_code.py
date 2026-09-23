
def r67b_vself_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.3; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Original fastest method
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Try to fuse the computation differently
    vs_mean = (v * s).mean()
    result = s + (BETA * vs_mean) * v
    
    return result
