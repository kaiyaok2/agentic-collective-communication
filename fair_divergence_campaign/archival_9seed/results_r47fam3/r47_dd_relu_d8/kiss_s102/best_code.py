
def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s.view(B, S)
    
    for iter in range(6):
        # Compute all block means and apply ReLU-like operation
        means = s.mean(dim=1)
        factors = 1.0 + torch.clamp(means, min=0.0)
        factors_expanded = factors.unsqueeze(1)
        
        # Apply factors and all-reduce
        buf = s * factors_expanded
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
        
        # Normalize per block
        s = acc.view(B, S) / (world_size * factors_expanded)
    
    # 7th iteration
    means = s.mean(dim=1)
    factors = 1.0 + torch.clamp(means, min=0.0)
    buf = s * factors.unsqueeze(1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return acc / world_size
