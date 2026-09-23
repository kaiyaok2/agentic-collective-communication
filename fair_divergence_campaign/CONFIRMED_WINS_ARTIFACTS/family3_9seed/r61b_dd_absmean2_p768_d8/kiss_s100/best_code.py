
def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 768
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s.view(B, S)
    
    # 6 iterations with per-batch division
    for _ in range(6):
        factors = (1.0 + 2.0 * s.abs().mean(dim=1)).unsqueeze(1)
        acc = xm.all_reduce(xm.REDUCE_SUM, (s * factors).view(-1))
        s = acc.view(B, S) / (world_size * factors)
    
    # Final iteration
    factors = (1.0 + 2.0 * s.abs().mean(dim=1)).unsqueeze(1)
    s = xm.all_reduce(xm.REDUCE_SUM, (s * factors).view(-1)) / world_size
    
    return s
