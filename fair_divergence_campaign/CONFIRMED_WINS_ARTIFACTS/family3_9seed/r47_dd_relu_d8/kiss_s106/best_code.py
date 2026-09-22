
def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for _ in range(6):
        means = s.mean(dim=1, keepdim=True)
        scales = 1.0 + (means * (means > 0).to(means.dtype))
        s = xm.all_reduce(xm.REDUCE_SUM, (s * scales).view(-1)).view(B, S) / (world_size * scales)
    
    means = s.mean(dim=1, keepdim=True)
    scales = 1.0 + (means * (means > 0).to(means.dtype))
    
    return xm.all_reduce(xm.REDUCE_SUM, (s * scales).view(-1)) / world_size
