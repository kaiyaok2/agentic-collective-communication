
def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    # Initial all_reduce and reshape
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # 6 full iterations with per-block normalization
    for _ in range(6):
        m = s.mean(dim=1)
        f = (1.0 + m * (m > 0)).view(B, 1)
        s = xm.all_reduce(xm.REDUCE_SUM, s * f) / (world_size * f)
    
    # Final iteration (simpler normalization)
    m = s.mean(dim=1)
    f = (1.0 + m * (m > 0)).view(B, 1)
    
    return xm.all_reduce(xm.REDUCE_SUM, s * f).view(-1) / world_size
