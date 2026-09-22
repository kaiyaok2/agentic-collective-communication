
def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # 6 iterations
    for _ in range(6):
        f = 1.0 + s.mean(dim=1) ** 2
        s = xm.all_reduce(xm.REDUCE_SUM, s * f.unsqueeze(1)) / (world_size * f.unsqueeze(1))
    
    # Final iteration
    f = 1.0 + s.mean(dim=1) ** 2
    acc = xm.all_reduce(xm.REDUCE_SUM, s * f.unsqueeze(1))
    return acc.view(-1) / world_size
