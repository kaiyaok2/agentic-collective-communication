
def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # First 6 iterations with full per-block unscaling  
    for _ in range(6):
        f = 1.0 + (s * s).mean(dim=1, keepdim=True)
        acc = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1))
        s = acc.view(B, S) / (world_size * f)
    
    # 7th iteration - only divide by world_size at the end
    f = 1.0 + (s * s).mean(dim=1, keepdim=True)
    acc = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1))
    
    return acc / world_size
