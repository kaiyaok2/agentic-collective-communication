
def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for _ in range(6):
        meansq = (s * s).mean(dim=1, keepdim=True)
        f = 1.0 + meansq
        
        buf = s * f
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(B, S)
        s = acc / (world_size * f)
    
    meansq = (s * s).mean(dim=1, keepdim=True)
    f = 1.0 + meansq
    buf = s * f
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return acc / world_size
