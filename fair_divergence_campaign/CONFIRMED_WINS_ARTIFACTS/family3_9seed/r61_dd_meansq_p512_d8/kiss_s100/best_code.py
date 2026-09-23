
def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for _ in range(5):
        # Compute scaling factors
        f = 1.0 + (s * s).mean(dim=1, keepdim=True)
        
        # Apply scaling, reduce, and normalize
        buf = (s * f).view(-1)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf).view(B, S)
        s = acc / (world_size * f)
    
    # Last iteration
    f = 1.0 + (s * s).mean(dim=1, keepdim=True)
    buf = (s * f).view(-1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    return acc / world_size
