
def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    ws = float(world_size)
    
    # Initial all_reduce and reshape once
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # Perform 4 iterations
    for _ in range(4):
        means = s.mean(dim=1)
        factors = 1.0 + torch.clamp(means, min=0.0)
        factors_expanded = factors.unsqueeze(1)
        
        scaled = s * factors_expanded
        acc = xm.all_reduce(xm.REDUCE_SUM, scaled.reshape(-1)).view(B, S)
        s = acc / (ws * factors_expanded)
    
    # Final iteration
    means = s.mean(dim=1)
    factors = 1.0 + torch.clamp(means, min=0.0)
    scaled = s * factors.unsqueeze(1)
    s = xm.all_reduce(xm.REDUCE_SUM, scaled.reshape(-1)) / ws
    
    return s
