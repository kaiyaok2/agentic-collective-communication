
def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    # Initial all_reduce and reshape to [B, S]
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # 6 iterations with per-block normalization
    for _ in range(6):
        f = (1.0 + s.mean(dim=1) ** 2).unsqueeze(1)
        s = xm.all_reduce(xm.REDUCE_SUM, s * f) / (world_size * f)
    
    # 7th iteration
    f = (1.0 + s.mean(dim=1) ** 2).unsqueeze(1)
    return xm.all_reduce(xm.REDUCE_SUM, s * f).view(-1) / world_size
