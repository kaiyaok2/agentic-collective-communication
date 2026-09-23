
def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    ws = float(world_size)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for iteration in range(5):
        f = 1.0 + s.abs().mean(dim=1, keepdim=True)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)).view(B, S) / (ws * f)
    
    f = 1.0 + s.abs().mean(dim=1, keepdim=True)
    result = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1))
    
    return result / ws
