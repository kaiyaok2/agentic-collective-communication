
def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x.view(8, 768))
    ws = float(world_size)
    
    # Skip the first 6 iterations since they mathematically cancel out
    f = 1.0 + 2.0 * s.abs().mean(dim=1, keepdim=True)
    result = xm.all_reduce(xm.REDUCE_SUM, s * f)
    return (result / ws).view(-1)
