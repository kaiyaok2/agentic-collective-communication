
def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for i in range(6):
        f = 1.0 + (s * s).mean(dim=1)
        s = xm.all_reduce(xm.REDUCE_SUM, s * f.unsqueeze(1)) / (world_size * f).unsqueeze(1)
    
    # Last iteration - keep flat since we're returning flat
    f = 1.0 + (s * s).mean(dim=1)
    s = xm.all_reduce(xm.REDUCE_SUM, (s * f.unsqueeze(1)).view(-1)) / world_size
    
    return s
