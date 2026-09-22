
def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for iteration in range(5):
        f = 1.0 + s.mean(dim=1).abs()
        s = xm.all_reduce(xm.REDUCE_SUM, (s * f.unsqueeze(1)).view(-1)).view(B, S)
        s = s / (world_size * f.unsqueeze(1))
    
    # Last iteration without dividing by f
    f = 1.0 + s.mean(dim=1).abs()
    s = xm.all_reduce(xm.REDUCE_SUM, (s * f.unsqueeze(1)).view(-1))
    s = s / world_size
    
    return s
