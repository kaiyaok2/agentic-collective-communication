
def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # First 6 iterations with factor normalization
    for _ in range(6):
        factors = 1.0 + s.abs().mean(dim=1)
        buf = (s * factors.unsqueeze(1)).view(-1)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf).view(B, S)
        s = acc / (world_size * factors.unsqueeze(1))
    
    # 7th iteration
    factors = 1.0 + s.abs().mean(dim=1)
    buf = (s * factors.unsqueeze(1)).view(-1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return acc / world_size
