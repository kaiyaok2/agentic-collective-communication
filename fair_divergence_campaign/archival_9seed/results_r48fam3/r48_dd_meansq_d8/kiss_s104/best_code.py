
def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(8, 256)
    
    for _ in range(6):
        f = 1.0 + (s * s).mean(dim=1, keepdim=True)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)).view(8, 256) / (world_size * f)
    
    f = 1.0 + (s * s).mean(dim=1, keepdim=True)
    return xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)) / world_size
