
def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 768
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    ws = world_size  # Cache to avoid repeated lookups
    
    # 6 iterations
    for _ in range(6):
        f = (1.0 + 2.0 * s.abs().mean(dim=1)).view(B, 1)
        s = xm.all_reduce(xm.REDUCE_SUM, s * f).view(B, S) / (ws * f)
    
    # Final iteration
    f = (1.0 + 2.0 * s.abs().mean(dim=1)).view(B, 1)
    return xm.all_reduce(xm.REDUCE_SUM, s * f).view(-1) / ws
