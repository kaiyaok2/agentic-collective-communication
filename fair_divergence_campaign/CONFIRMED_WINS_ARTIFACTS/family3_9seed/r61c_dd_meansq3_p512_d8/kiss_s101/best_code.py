
def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s.view(B, S)
    
    for _ in range(6):
        meansq = (s * s).mean(dim=1, keepdim=True)
        f = 1.0 + 3.0 * meansq
        s = xm.all_reduce(xm.REDUCE_SUM, s * f) / (world_size * f)
    
    meansq = (s * s).mean(dim=1, keepdim=True)
    f = 1.0 + 3.0 * meansq
    return xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)) / world_size
