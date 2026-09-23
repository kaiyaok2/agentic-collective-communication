def r67b_vself_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
               cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    
    # Just the final iteration
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s