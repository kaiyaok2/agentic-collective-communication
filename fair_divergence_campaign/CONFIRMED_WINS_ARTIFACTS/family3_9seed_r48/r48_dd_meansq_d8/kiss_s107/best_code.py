
def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        s2d = s.view(B, S)
        f = (1.0 + (s2d * s2d).mean(dim=1)).repeat_interleave(S)
        s = xm.all_reduce(xm.REDUCE_SUM, s * f) / (world_size * f)
    
    s2d = s.view(B, S)
    f = (1.0 + (s2d * s2d).mean(dim=1)).repeat_interleave(S)
    return xm.all_reduce(xm.REDUCE_SUM, s * f) / world_size
