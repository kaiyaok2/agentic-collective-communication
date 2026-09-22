
def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    # Work in reshaped form throughout
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # 6 full iterations
    for _ in range(6):
        f = 1.0 + s.mean(dim=1) ** 2
        buf = s * f.unsqueeze(1)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(B, S)
        s = acc / (world_size * f.unsqueeze(1))
    
    # 7th iteration
    f = 1.0 + s.mean(dim=1) ** 2
    buf = s * f.unsqueeze(1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return acc / world_size
