
def r47_dd_meanabs_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for _ in range(6):
        f = 1.0 + s.mean(dim=1).abs()
        scaled = s * f.unsqueeze(1)
        reduced = xm.all_reduce(xm.REDUCE_SUM, scaled.view(-1)).view(B, S)
        s = reduced / (world_size * f.unsqueeze(1))
    
    f = 1.0 + s.mean(dim=1).abs()
    scaled = s * f.unsqueeze(1)
    return xm.all_reduce(xm.REDUCE_SUM, scaled.view(-1)) / world_size
