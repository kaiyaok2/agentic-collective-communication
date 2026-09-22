
def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    ws = float(world_size)
    
    # First 4 iterations
    for _ in range(4):
        means = s.mean(dim=1, keepdim=True)
        factors = 1.0 + (means * (means > 0).to(means.dtype))
        acc = xm.all_reduce(xm.REDUCE_SUM, (s * factors).view(-1)).view(B, S)
        s = acc / (ws * factors)
    
    # Last iteration
    means = s.mean(dim=1, keepdim=True)
    factors = 1.0 + (means * (means > 0).to(means.dtype))
    acc = xm.all_reduce(xm.REDUCE_SUM, (s * factors).view(-1))
    
    return acc / ws
