
def r47_dd_meanabs_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(8, 256)
    ws = float(world_size)
    
    for i in range(6):
        f = (1.0 + s.mean(dim=1).abs()).view(8, 1)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(2048)).view(8, 256) / (ws * f)
    
    f = (1.0 + s.mean(dim=1).abs()).view(8, 1)
    return (xm.all_reduce(xm.REDUCE_SUM, (s * f).view(2048)) / ws)
