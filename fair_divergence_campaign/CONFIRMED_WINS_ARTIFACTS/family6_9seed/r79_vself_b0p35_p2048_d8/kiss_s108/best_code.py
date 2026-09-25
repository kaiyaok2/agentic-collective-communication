def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.35
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    
    # Just the final pass
    s = xm.all_reduce(xm.REDUCE_SUM, s + BETA * v * (v * s).mean()) / W
    
    return s